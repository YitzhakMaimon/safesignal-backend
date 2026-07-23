#!/usr/bin/env bash
# Deploys services/rag_service/ to the standalone EC2 instance and runs the
# full end-to-end testing protocol against the live cloud endpoint.
#
# Run this FROM THE PROJECT ROOT, on a machine whose network allows outbound
# SSH (this can't run from a network that resets non-HTTPS outbound traffic).
#
# WHY THERE ARE NO AWS CREDENTIALS IN THIS SCRIPT: the EC2 instance already
# has an IAM instance role (safesignal-rag-ssm-role, attached as instance
# profile safesignal-rag-ssm-profile) with AmazonBedrockFullAccess. boto3
# inside the container will pick up temporary credentials automatically via
# the instance metadata service -- no static keys need to leave your machine.
# (Instance metadata hop limit was already confirmed at 2, which is required
# for a Docker container -- one network namespace deeper than the host -- to
# reach the metadata service at all.)
set -euo pipefail

EC2_IP="3.89.7.120"
EC2_USER="ubuntu"
KEY_FILE="safesignal-shared-key.pem"
REMOTE_DIR="/home/ubuntu/rag_service"
BASE_URL="http://${EC2_IP}:8001"

chmod 400 "$KEY_FILE"

echo "=== Step 1: SSH connectivity check ==="
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
  "${EC2_USER}@${EC2_IP}" "echo SSH_OK && cat /etc/os-release | head -3"

echo ""
echo "=== Step 2: Installing Docker (if not already present) ==="
ssh -i "$KEY_FILE" "${EC2_USER}@${EC2_IP}" bash -s <<'REMOTE_SCRIPT'
set -e
if ! command -v docker &> /dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
fi
sudo docker --version
REMOTE_SCRIPT

echo ""
echo "=== Step 3: Transferring services/rag_service/ (main.py, requirements.txt, Dockerfile, training/ incl. FAISS index) ==="
ssh -i "$KEY_FILE" "${EC2_USER}@${EC2_IP}" "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR}"
# -p preserves the original file mtimes -- matters because rag_core.py's FAISS
# cache validity check is mtime-based (see services/rag_service/training/faiss_index/
# source_meta.json), and preserving the exact mtime this file already has avoids
# an unnecessary, costly full re-embed of the 4992-document corpus via Bedrock
# on first deploy. If the timestamps don't line up perfectly, the container will
# just rebuild the index once (a few real minutes + Bedrock calls) -- not a bug,
# a known limitation of the mtime-based cache we found while testing locally.
scp -i "$KEY_FILE" -p -r services/rag_service/* "${EC2_USER}@${EC2_IP}:${REMOTE_DIR}/"

echo ""
echo "=== Step 4: Environment variables (no secrets -- see note above; only non-secret config) ==="
echo "Using BEDROCK_AWS_REGION=us-east-1 (credentials come from the instance's IAM role, not from .env)"

echo ""
echo "=== Step 5: Building and running the container on the EC2 instance ==="
ssh -i "$KEY_FILE" "${EC2_USER}@${EC2_IP}" bash -s <<REMOTE_SCRIPT
set -e
cd ${REMOTE_DIR}
sudo docker build -t safesignal-rag-service .
sudo docker rm -f rag_service 2>/dev/null || true
sudo docker run -d --name rag_service -p 8001:8001 \
  -e BEDROCK_AWS_REGION=us-east-1 \
  --restart unless-stopped \
  safesignal-rag-service
REMOTE_SCRIPT

echo ""
echo "=== Step 6: Architectural independence verification ==="
ssh -i "$KEY_FILE" "${EC2_USER}@${EC2_IP}" bash -s <<'REMOTE_SCRIPT'
echo "--- docker ps (should show exactly one container: rag_service) ---"
sudo docker ps
echo ""
echo "--- listening ports on this host (ss -tlnp; expect 22 for sshd + 8001 for our container, nothing else app-related) ---"
sudo ss -tlnp
echo ""
echo "--- container's own process tree (should be uvicorn/python only -- no other local service processes) ---"
sudo docker top rag_service
echo ""
echo "--- outbound connections FROM the container while idle (should be none -- it only talks to AWS Bedrock on demand, not to any other local service) ---"
sudo docker exec rag_service sh -c "command -v ss >/dev/null && ss -tnp || echo '(no ss in slim image -- expected, not an error)'"
REMOTE_SCRIPT

echo ""
echo "=== Waiting for the service to become ready (first boot may rebuild the FAISS index -- see Step 3 note) ==="
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${BASE_URL}/docs" 2>/dev/null || echo "000")
  echo "attempt $i: HTTP $code"
  [ "$code" = "200" ] && break
  sleep 10
done

echo ""
echo "=== Step 7: End-to-end testing protocol against the LIVE cloud endpoint ==="

echo ""
echo "--- Test 1: Hebrew Distress ---"
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" \
  -d '{"description": "אני מרגיש אבוד ואין לי למי לפנות, חושב לפגוע בעצמי"}'
echo ""

echo "--- Test 2: Hebrew Nuance (teen slang / indirect cry for help) ---"
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" \
  -d '{"description": "נמאס לי מהמשחק הזה, אני שקוף לכולם ואף אחד לא היה שם לב אם הייתי נעלם"}'
echo ""

echo "--- Test 3: English Distress ---"
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" \
  -d '{"description": "I feel completely hopeless and lonely, I need urgent help"}'
echo ""

echo "--- Test 4: English Nuance (internet slang) ---"
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" \
  -d '{"description": "not gonna lie ngl I kinda just want to disappear fr fr, nobody would even notice"}'
echo ""

echo "--- Test 5a: Empty string (valid type, NOT expected to 422 -- retrieve() handles it as 'no context', check the actual code path) ---"
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" -d '{"description": ""}'
echo ""
echo "--- Test 5b: Missing required field (THIS is the real malformed-payload case -- expect HTTP 422) ---"
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST "${BASE_URL}/query" -H "Content-Type: application/json" -d '{}'

echo ""
echo "--- Test 6: Out-of-Domain Query ---"
echo "NOTE: rag_service has NO confidence-threshold gating in its current implementation --"
echo "it always returns the top-3 nearest FAISS vectors, however irrelevant. Watch the"
echo "similarity_score values below rather than expecting a distinct 'low confidence' field."
curl -s -X POST "${BASE_URL}/query" -H "Content-Type: application/json; charset=utf-8" \
  -d '{"description": "How do I bake a chocolate cake?"}'
echo ""

echo ""
echo "=== DONE. Paste the full output back so the results can be verified. ==="
