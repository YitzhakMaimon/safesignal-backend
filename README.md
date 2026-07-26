# SafeSignal

## מה להריץ אחרי כיבוי/הפעלה מחדש של המחשב

הכי פשוט: לפתוח שיחה עם Claude Code ולבקש "תעלה הכל מחדש" — כל השלבים למטה לא דורשים הרשאות אדמין, ואפשר להריץ אותם דרכו.

**חשוב**: זה *לא* עולה אוטומטית לבד. ניסינו להפוך את זה ל-Windows Scheduled Tasks קבועים (2026-07-26), אבל משהו במחשב הזה (כנראה סוכן האבטחה הארגוני של Amdocs, Zscaler) הורג תהליכים שמשוגרים דרך Task Scheduler תוך כ-20-90 שניות מהרגע שהם נהיים "פעילים ברשת" (פותחים פורט/session) — בעוד שאותם תהליכים בדיוק, כשמריצים אותם ידנית (ad-hoc), שרדו שעות בלי בעיה. לכן חוזרים כל פעם על ההרצה הידנית במקום לסמוך על persistence אוטומטי.

### הסדר המדויק

1. **Docker Desktop** — לא צריך לעשות כלום, מוגדר לעלות אוטומטית בהתחברות. n8n וכל שאר הקונטיינרים המקומיים עולים איתו.

2. **הבקאנד (`safesignal.py`)** — טעינת מודלים (HeBERT + RAG index) לוקחת דקה-שתיים:
   ```bash
   PYTHONIOENCODING=utf-8 nohup ./.venv/Scripts/python.exe safesignal.py > scripts/backend.log 2>&1 &
   disown
   ```

3. **3 מנהרות SSM** (מקומי → EC2, לכל אחת EC2 שונה — ראו הערות בראש כל סקריפט):
   ```powershell
   Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\ssm-tunnel-loop.ps1" -WindowStyle Hidden
   Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\ssm-tunnel-rag-service-loop.ps1" -WindowStyle Hidden
   Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\ssm-tunnel-output-screening-loop.ps1" -WindowStyle Hidden
   ```

4. **מנהרת ה-reverse tunnel** (EC2 langgraph-agent → המחשב הזה, לטלגרם ול-alerts/review-queue) — **רק אחרי** ששלב 3 הסתיים, לא בו-זמנית איתו. שתיהן פונות לאותו EC2 instance (`i-0e10488a3bbbefef9`), ומצאנו שהוא לא סופג יפה יותר מ-2 sessions בו-זמנית — אם שתי המנהרות האלה מתחילות ביחד, אחת מהן עלולה ליפול:
   ```powershell
   Start-Process powershell.exe -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts\ssh-reverse-tunnel-loop.ps1" -WindowStyle Hidden
   ```

5. **`telegram_bridge.py` על ה-EC2** — **אין צורך לגעת בו**. הוא רץ שם עצמאית ולא מושפע מכיבוי/הפעלה של המחשב המקומי בכלל.

### אזהרה תפעולית

אחרי שהמנהרות למעלה חיות, **להימנע מלפתוח session אבחוני/ידני נוסף** ל-EC2 `langgraph-agent` (`i-0e10488a3bbbefef9`) — פתיחת session שלישי לאותו instance בזמן שהמנהרות רצות עלולה להפיל אחת מהן (נצפה ב-2026-07-26).

### בדיקת תקינות מהירה

```bash
netstat -ano | grep -E ":8000 |:18001 |:18002 |:18006 " | grep LISTENING
tasklist //FI "IMAGENAME eq ssh.exe"
```

כל 4 השורות (8000/18001/18002/18006) אמורות להופיע כ-LISTENING, ו-`ssh.exe` אמור להופיע פעם אחת.
