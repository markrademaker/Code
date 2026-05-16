# Platja Reservations

Reservation site for our family villa in Mas Nou, near Platja d'Aro on the
Costa Brava. Visitors can read about the villa, see which dates are already
booked, and submit a booking request that lands in the owners' inboxes.

## Stack

- **Next.js 14** (App Router, TypeScript)
- **Tailwind CSS** for styling
- **Resend** for transactional email
- **date-fns** for date math
- **zod** for request validation

Booked date ranges live in `data/bookings.json` — edit that file (and redeploy)
when a new booking is confirmed.

## Local setup

```bash
cp .env.example .env.local      # fill in RESEND_API_KEY
npm install
npm run dev
```

Open http://localhost:3000.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `RESEND_API_KEY` | API key from [resend.com](https://resend.com) |
| `BOOKING_FROM_EMAIL` | Verified sender (use `onboarding@resend.dev` for dev) |
| `BOOKING_TO_EMAILS` | Comma-separated recipients (defaults to `neleman3@gmail.com,rademair@gmail.com`) |

## Managing availability

Confirmed bookings are stored as date ranges in `data/bookings.json`:

```json
{
  "bookings": [
    { "id": "smith-jul-2026", "start": "2026-07-15", "end": "2026-07-29" }
  ]
}
```

When a request comes in by email and you accept it, add a new entry, commit,
and redeploy.

## Booking flow

1. Visitor lands on the homepage, browses the villa overview and availability
   calendar.
2. They fill in the booking form with dates, guests, and a message.
3. `POST /api/book` validates the input and checks availability against
   `bookings.json`.
4. If the dates are free, a request email is sent via Resend to the recipients
   in `BOOKING_TO_EMAILS` with the visitor as `reply-to`.

## Deploy

Easiest path: push to GitHub and import the project into Vercel. Add the env
vars in the Vercel dashboard. The static JSON file is read at build/runtime —
just trigger a redeploy after editing `data/bookings.json`.
