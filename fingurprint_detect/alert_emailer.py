"""
==============================================================================
  FINGERPRINT ATTENDANCE VIOLATION — EMAIL ALERT MODULE
==============================================================================
  Sends an HTML alert email with a snapshot image attached whenever an
  unpunched entry (VIOLATION) is detected by the front door AI monitor.

  Config is loaded from the parent folder's .env file:
    ALERT_EMAIL_SENDER       — Gmail / Outlook address that sends the alert
    ALERT_EMAIL_PASSWORD     — App Password (Gmail) or account password
    ALERT_EMAIL_RECIPIENT    — Comma-separated recipient address(es)
    ALERT_EMAIL_SMTP         — SMTP host  (default: smtp.gmail.com)
    ALERT_EMAIL_PORT         — SMTP port  (default: 587)
    ALERT_EMAIL_COOLDOWN_SEC — Min seconds between emails (default: 300)

  Usage:
    from alert_emailer import ViolationEmailer
    emailer = ViolationEmailer(camera_name="CAMERA1")
    emailer.send(snapshot_path="/path/to/snap.jpg", track_id=7,
                 timestamp="2026-08-14 10:38:05")
==============================================================================
"""

import os
import time
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# Load .env from parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()  # Also check local .env


class ViolationEmailer:
    """
    Thread-safe, cooldown-aware email alerter for fingerprint violations.
    All sends happen in a daemon thread so the video feed is never blocked.
    """

    def __init__(self, camera_name: str = "CAMERA1"):
        self.camera_name = camera_name

        # SMTP credentials from environment
        self.sender      = os.environ.get("ALERT_EMAIL_SENDER", "").strip()
        self.password    = os.environ.get("ALERT_EMAIL_PASSWORD", "").strip()
        self.recipient   = os.environ.get("ALERT_EMAIL_RECIPIENT", "").strip()
        self.smtp_host   = os.environ.get("ALERT_EMAIL_SMTP", "smtp.gmail.com").strip()
        self.smtp_port   = int(os.environ.get("ALERT_EMAIL_PORT", "587"))
        self.cooldown    = float(os.environ.get("ALERT_EMAIL_COOLDOWN_SEC", "300"))

        self._last_sent  = 0.0          # epoch timestamp of last successful send
        self._lock       = threading.Lock()
        self._enabled    = bool(self.sender and self.password and self.recipient)

        if self._enabled:
            print(f"[EMAIL ALERT] Configured → {self.sender} → {self.recipient} "
                  f"| Cooldown: {int(self.cooldown)}s")
        else:
            print("[EMAIL ALERT] ⚠  Not configured — set ALERT_EMAIL_* vars in .env to enable alerts.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, snapshot_path: str, track_id: int, timestamp: str):
        """
        Trigger a violation alert email (non-blocking — fires in background thread).
        Respects the cooldown period to avoid flooding the inbox.
        """
        if not self._enabled:
            return

        with self._lock:
            elapsed = time.time() - self._last_sent
            if elapsed < self.cooldown:
                remaining = int(self.cooldown - elapsed)
                print(f"[EMAIL ALERT] Cooldown active — next alert in {remaining}s")
                return
            # Reserve the slot immediately (even before the thread sends)
            self._last_sent = time.time()

        thread = threading.Thread(
            target=self._send_worker,
            args=(snapshot_path, track_id, timestamp),
            daemon=True,
            name=f"EmailAlert-T{track_id}"
        )
        thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_worker(self, snapshot_path: str, track_id: int, timestamp: str):
        """Runs in background thread — builds and sends the MIME email."""
        try:
            msg = self._build_message(snapshot_path, track_id, timestamp)
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.login(self.sender, self.password)
                recipients = [r.strip() for r in self.recipient.split(",")]
                server.sendmail(self.sender, recipients, msg.as_string())
            print(f"[EMAIL ALERT] ✅ Violation alert sent → {self.recipient} (Track {track_id})")
        except smtplib.SMTPAuthenticationError:
            print("[EMAIL ALERT] ❌ Auth failed — check ALERT_EMAIL_PASSWORD in .env "
                  "(Gmail requires an App Password, not your account password).")
        except smtplib.SMTPException as e:
            print(f"[EMAIL ALERT] ❌ SMTP error: {e}")
        except Exception as e:
            print(f"[EMAIL ALERT] ❌ Unexpected error sending email: {e}")

    def _build_message(self, snapshot_path: str, track_id: int, timestamp: str) -> MIMEMultipart:
        """Constructs an HTML email with the snapshot JPEG attached inline."""
        recipients = [r.strip() for r in self.recipient.split(",")]

        msg = MIMEMultipart("related")
        msg["Subject"] = f"🚨 UNPUNCHED ENTRY ALERT — {self.camera_name} | {timestamp}"
        msg["From"]    = f"CCTV Monitor <{self.sender}>"
        msg["To"]      = ", ".join(recipients)

        # ── HTML body ──────────────────────────────────────────────────
        has_image = os.path.isfile(snapshot_path)
        img_tag   = '<img src="cid:snapshot" style="width:100%;max-width:720px;border-radius:8px;" />' \
                    if has_image else "<p><em>(Snapshot not available)</em></p>"

        html = f"""
        <html>
        <body style="margin:0;padding:0;background:#0f172a;font-family:Arial,sans-serif;color:#e2e8f0;">
          <table width="100%" cellpadding="0" cellspacing="0" style="max-width:720px;margin:30px auto;">
            <tr>
              <td style="background:#1e293b;border-radius:12px 12px 0 0;padding:24px 32px;">
                <h1 style="margin:0;font-size:22px;color:#f87171;">
                  🚨 Unpunched Entry Detected
                </h1>
                <p style="margin:4px 0 0;color:#94a3b8;font-size:13px;">
                  Biometric fingerprint compliance violation
                </p>
              </td>
            </tr>
            <tr>
              <td style="background:#1e293b;padding:0 32px 24px;">
                <table cellpadding="8" style="width:100%;background:#0f172a;border-radius:8px;margin:16px 0;">
                  <tr>
                    <td style="color:#94a3b8;font-size:13px;width:130px;">📷 Camera</td>
                    <td style="color:#f1f5f9;font-weight:bold;">{self.camera_name}</td>
                  </tr>
                  <tr style="background:#1e293b;">
                    <td style="color:#94a3b8;font-size:13px;">🕐 Time</td>
                    <td style="color:#f1f5f9;font-weight:bold;">{timestamp}</td>
                  </tr>
                  <tr>
                    <td style="color:#94a3b8;font-size:13px;">🆔 Track ID</td>
                    <td style="color:#f1f5f9;font-weight:bold;">{track_id}</td>
                  </tr>
                  <tr style="background:#1e293b;">
                    <td style="color:#94a3b8;font-size:13px;">⚠️ Violation</td>
                    <td style="color:#fb923c;font-weight:bold;">Entered without fingerprint scan</td>
                  </tr>
                </table>
                <p style="color:#94a3b8;font-size:12px;margin:4px 0 12px;">
                  Snapshot captured at time of violation:
                </p>
                {img_tag}
              </td>
            </tr>
            <tr>
              <td style="background:#0f172a;border-radius:0 0 12px 12px;padding:16px 32px;
                         text-align:center;color:#475569;font-size:11px;">
                GWC CCTV Monitor · Fingerprint Attendance AI · Auto-generated alert
              </td>
            </tr>
          </table>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, "html"))

        # ── Attach snapshot inline ─────────────────────────────────────
        if has_image:
            try:
                with open(snapshot_path, "rb") as f:
                    img_data = f.read()
                img_part = MIMEImage(img_data, _subtype="jpeg")
                img_part.add_header("Content-ID", "<snapshot>")
                img_part.add_header(
                    "Content-Disposition", "inline",
                    filename=os.path.basename(snapshot_path)
                )
                msg.attach(img_part)
            except Exception as e:
                print(f"[EMAIL ALERT] Could not attach snapshot: {e}")

        return msg


# ------------------------------------------------------------------
# Quick standalone test  →  python alert_emailer.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== ViolationEmailer Self-Test ===")
    emailer = ViolationEmailer(camera_name="CAMERA1")
    if not emailer._enabled:
        print("Set ALERT_EMAIL_* vars in .env first, then re-run.")
    else:
        emailer.send(
            snapshot_path="",
            track_id=99,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        time.sleep(5)   # wait for background thread
        print("Test complete.")
