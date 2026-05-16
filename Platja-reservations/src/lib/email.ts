import { Resend } from "resend";
import type { Booking, BookingStatus } from "@prisma/client";
import { signActionToken } from "@/lib/tokens";
import { format } from "date-fns";

function getRecipients(): string[] {
  const raw = process.env.BOOKING_TO_EMAILS ?? "neleman3@gmail.com,rademair@gmail.com";
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function getBaseUrl(): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ??
    process.env.NEXT_PUBLIC_APP_BASE_URL?.replace(/\/$/, "") ??
    "http://localhost:3000"
  );
}

function escape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmt(d: Date): string {
  return format(d, "EEE d MMM yyyy");
}

function row(label: string, value: string): string {
  return `<tr><td style="padding:4px 12px 4px 0;color:#5a6b76">${label}</td><td style="padding:4px 0;color:#1f4a5f"><strong>${value}</strong></td></tr>`;
}

function button(href: string, label: string, bg: string): string {
  return `<a href="${href}" style="display:inline-block;padding:12px 22px;margin-right:8px;background:${bg};color:#fff;text-decoration:none;border-radius:9999px;font-weight:600">${label}</a>`;
}

function resend(): Resend {
  const key = process.env.RESEND_API_KEY;
  if (!key) throw new Error("RESEND_API_KEY is not configured");
  return new Resend(key);
}

function from(): string {
  const email = process.env.BOOKING_FROM_EMAIL ?? "onboarding@resend.dev";
  return `Villa Mas Nou <${email}>`;
}

export async function sendAdminBookingRequest(booking: Booking): Promise<void> {
  const base = getBaseUrl();
  const accept = `${base}/api/booking-action?t=${signActionToken(booking.id, "accept")}`;
  const tentative = `${base}/api/booking-action?t=${signActionToken(booking.id, "tentative")}`;
  const decline = `${base}/api/booking-action?t=${signActionToken(booking.id, "decline")}`;
  const adminUrl = `${base}/admin`;

  const html = `
    <h2 style="font-family:Georgia,serif;color:#1f4a5f">New booking request</h2>
    <p>${escape(booking.name)} has requested the villa for <strong>${fmt(booking.checkIn)} → ${fmt(booking.checkOut)}</strong>.</p>
    <table style="border-collapse:collapse;font-family:system-ui,sans-serif;font-size:14px;margin:8px 0 16px">
      ${row("Guest", escape(booking.name))}
      ${row("Email", escape(booking.email))}
      ${row("Phone", escape(booking.phone ?? "—"))}
      ${row("Guests", String(booking.guests))}
      ${row("Check-in", fmt(booking.checkIn))}
      ${row("Check-out", fmt(booking.checkOut))}
    </table>
    ${booking.message ? `<p style="background:#f5efe6;padding:12px 16px;border-radius:8px;font-family:system-ui,sans-serif;white-space:pre-wrap">${escape(booking.message)}</p>` : ""}
    <p style="margin-top:24px">
      ${button(accept, "✓ Accept", "#1f4a5f")}
      ${button(tentative, "⏳ Tentative", "#c97b5b")}
      ${button(decline, "✗ Decline", "#7a3737")}
    </p>
    <p style="font-size:13px;color:#5a6b76">Or open the <a href="${adminUrl}">admin dashboard</a> to view the full planning. Action links expire in 14 days.</p>
  `;

  const { error } = await resend().emails.send({
    from: from(),
    to: getRecipients(),
    replyTo: booking.email,
    subject: `Booking request: ${fmt(booking.checkIn)} → ${fmt(booking.checkOut)} (${booking.name})`,
    html,
  });
  if (error) throw new Error(`Resend (admin email): ${error.message}`);
}

export async function sendGuestStatusUpdate(
  booking: Booking,
  previousStatus: BookingStatus | null,
): Promise<void> {
  const messages: Record<BookingStatus, { subject: string; intro: string }> = {
    PENDING: {
      subject: "We received your booking request",
      intro: "Thanks for your request — we'll review it and get back to you soon.",
    },
    CONFIRMED: {
      subject: "Your booking is confirmed",
      intro: "Great news — your stay at Villa Mas Nou is confirmed. We're looking forward to having you.",
    },
    TENTATIVE: {
      subject: "Your booking is tentatively held",
      intro: "We've put a tentative hold on your dates. We'll get back to you shortly to confirm.",
    },
    DECLINED: {
      subject: "About your booking request",
      intro: "Unfortunately we aren't able to accommodate these dates. Please feel free to try other dates.",
    },
    CANCELLED: {
      subject: "Your booking has been cancelled",
      intro: "Your booking has been cancelled. If this was unexpected, please reply to this email.",
    },
  };

  const m = messages[booking.status];
  const movedHint = previousStatus === booking.status
    ? "Your dates have been updated."
    : "";

  const html = `
    <h2 style="font-family:Georgia,serif;color:#1f4a5f">Hi ${escape(booking.name)},</h2>
    <p>${escape(m.intro)} ${movedHint ? `<br><em>${escape(movedHint)}</em>` : ""}</p>
    <table style="border-collapse:collapse;font-family:system-ui,sans-serif;font-size:14px;margin:8px 0 16px">
      ${row("Check-in", fmt(booking.checkIn))}
      ${row("Check-out", fmt(booking.checkOut))}
      ${row("Guests", String(booking.guests))}
      ${row("Status", booking.status.toLowerCase())}
    </table>
    <p>— Villa Mas Nou, Platja d'Aro</p>
  `;

  const { error } = await resend().emails.send({
    from: from(),
    to: booking.email,
    replyTo: getRecipients()[0],
    subject: m.subject,
    html,
  });
  if (error) throw new Error(`Resend (guest email): ${error.message}`);
}
