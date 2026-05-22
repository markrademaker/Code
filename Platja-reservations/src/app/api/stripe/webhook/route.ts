import { NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { stripe, isStripeConfigured } from "@/lib/stripe";
import { sendGuestStatusUpdate } from "@/lib/email";
import type Stripe from "stripe";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  if (!isStripeConfigured()) {
    return NextResponse.json({ error: "Stripe not configured" }, { status: 503 });
  }
  const secret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!secret) {
    return NextResponse.json({ error: "Webhook secret not configured" }, { status: 503 });
  }

  const signature = req.headers.get("stripe-signature");
  const body = await req.text();
  if (!signature) {
    return NextResponse.json({ error: "Missing signature" }, { status: 400 });
  }

  let event: Stripe.Event;
  try {
    event = stripe().webhooks.constructEvent(body, signature, secret);
  } catch (err) {
    console.error("stripe webhook signature verification failed", err);
    return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
  }

  try {
    if (event.type === "checkout.session.completed") {
      const session = event.data.object as Stripe.Checkout.Session;
      const bookingId = session.metadata?.bookingId ?? session.client_reference_id;
      if (bookingId && session.payment_status === "paid") {
        const booking = await prisma.booking.findUnique({ where: { id: bookingId } });
        if (booking && booking.paymentStatus !== "PAID") {
          const updated = await prisma.booking.update({
            where: { id: bookingId },
            data: {
              paymentStatus: "PAID",
              paidAt: new Date(),
              stripePaymentId:
                typeof session.payment_intent === "string"
                  ? session.payment_intent
                  : (session.payment_intent?.id ?? null),
            },
          });
          // best-effort confirmation email
          try {
            await sendGuestStatusUpdate(updated, booking.status);
          } catch {
            // ignore email failure
          }
        }
      }
    }

    if (event.type === "checkout.session.async_payment_succeeded") {
      const session = event.data.object as Stripe.Checkout.Session;
      const bookingId = session.metadata?.bookingId ?? session.client_reference_id;
      if (bookingId) {
        await prisma.booking
          .update({
            where: { id: bookingId },
            data: {
              paymentStatus: "PAID",
              paidAt: new Date(),
              stripePaymentId:
                typeof session.payment_intent === "string"
                  ? session.payment_intent
                  : (session.payment_intent?.id ?? null),
            },
          })
          .catch(() => null);
      }
    }

    if (event.type === "checkout.session.async_payment_failed") {
      const session = event.data.object as Stripe.Checkout.Session;
      const bookingId = session.metadata?.bookingId ?? session.client_reference_id;
      if (bookingId) {
        await prisma.booking
          .update({ where: { id: bookingId }, data: { paymentStatus: "UNPAID" } })
          .catch(() => null);
      }
    }

    return NextResponse.json({ received: true });
  } catch (err) {
    console.error("stripe webhook handler failed", err);
    return NextResponse.json({ error: "Handler failed" }, { status: 500 });
  }
}
