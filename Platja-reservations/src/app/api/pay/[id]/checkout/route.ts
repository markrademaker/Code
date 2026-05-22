import { NextResponse } from "next/server";
import { format } from "date-fns";
import { prisma } from "@/lib/db";
import { getCurrentUser } from "@/lib/user";
import { stripe, isStripeConfigured } from "@/lib/stripe";

export const runtime = "nodejs";

function getBaseUrl(req: Request): string {
  return (
    process.env.APP_BASE_URL?.replace(/\/$/, "") ?? new URL(req.url).origin
  );
}

export async function POST(req: Request, { params }: { params: { id: string } }) {
  if (!isStripeConfigured()) {
    return NextResponse.json(
      { error: "Online payment is not configured yet" },
      { status: 503 },
    );
  }

  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  const booking = await prisma.booking.findUnique({ where: { id: params.id } });
  if (!booking || booking.userId !== user.id) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  if (!booking.totalAmountCents || booking.totalAmountCents <= 0) {
    return NextResponse.json(
      { error: "No amount set on this booking yet" },
      { status: 400 },
    );
  }
  if (booking.paymentStatus === "PAID") {
    return NextResponse.json({ error: "Already paid" }, { status: 400 });
  }

  const base = getBaseUrl(req);
  try {
    const session = await stripe().checkout.sessions.create({
      mode: "payment",
      payment_method_types: ["card", "ideal", "paypal"],
      customer_email: user.email,
      client_reference_id: booking.id,
      line_items: [
        {
          price_data: {
            currency: "eur",
            product_data: {
              name: "Villa Mas Nou, Platja d'Aro",
              description: `${format(booking.checkIn, "EEE d MMM yyyy")} → ${format(booking.checkOut, "EEE d MMM yyyy")} · ${booking.guests} guests`,
            },
            unit_amount: booking.totalAmountCents,
          },
          quantity: 1,
        },
      ],
      metadata: {
        bookingId: booking.id,
        userId: user.id,
      },
      success_url: `${base}/my-bookings?paid=${booking.id}`,
      cancel_url: `${base}/pay/${booking.id}?cancelled=1`,
    });

    if (!session.url) {
      return NextResponse.json({ error: "Stripe didn't return a URL" }, { status: 500 });
    }

    await prisma.booking.update({
      where: { id: booking.id },
      data: { stripeSessionId: session.id },
    });

    return NextResponse.json({ url: session.url });
  } catch (err) {
    console.error("stripe checkout failed", err);
    return NextResponse.json(
      { error: "Could not start payment, please try again" },
      { status: 500 },
    );
  }
}
