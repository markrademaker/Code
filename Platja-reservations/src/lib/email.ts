import { Resend } from "resend";

export type BookingRequest = {
  name: string;
  email: string;
  phone?: string;
  checkIn: string;
  checkOut: string;
  guests: number;
  message?: string;
};

function getRecipients(): string[] {
  const raw = process.env.BOOKING_TO_EMAILS ?? "neleman3@gmail.com,rademair@gmail.com";
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function renderHtml(req: BookingRequest): string {
  return `
    <h2>New booking request — Villa Mas Nou</h2>
    <p>A new reservation request has been submitted for the villa near Platja d'Aro.</p>
    <table style="border-collapse:collapse">
      <tr><td><strong>Name</strong></td><td>${escape(req.name)}</td></tr>
      <tr><td><strong>Email</strong></td><td>${escape(req.email)}</td></tr>
      <tr><td><strong>Phone</strong></td><td>${escape(req.phone ?? "—")}</td></tr>
      <tr><td><strong>Check-in</strong></td><td>${escape(req.checkIn)}</td></tr>
      <tr><td><strong>Check-out</strong></td><td>${escape(req.checkOut)}</td></tr>
      <tr><td><strong>Guests</strong></td><td>${req.guests}</td></tr>
    </table>
    <h3>Message</h3>
    <p style="white-space:pre-wrap">${escape(req.message ?? "")}</p>
  `;
}

function escape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function sendBookingEmail(req: BookingRequest): Promise<void> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) throw new Error("RESEND_API_KEY is not configured");

  const resend = new Resend(apiKey);
  const from = process.env.BOOKING_FROM_EMAIL ?? "onboarding@resend.dev";

  const { error } = await resend.emails.send({
    from: `Villa Mas Nou <${from}>`,
    to: getRecipients(),
    replyTo: req.email,
    subject: `Booking request: ${req.checkIn} → ${req.checkOut} (${req.name})`,
    html: renderHtml(req),
  });

  if (error) throw new Error(error.message);
}
