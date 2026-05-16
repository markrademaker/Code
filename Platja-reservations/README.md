# Platja Reservations

Reservation site for our family villa in Mas Nou, near Platja d'Aro on the
Costa Brava.

- Public site: villa overview, photos page, weather page (live forecast +
  monthly climate), availability calendar, and a booking request form.
- Booking requests are stored in Postgres and emailed to the owners with
  **Accept / Tentative / Decline** action buttons.
- `/admin` shows the full planning, lists every booking, and lets the owners
  move dates, change status, or cancel — the guest is auto-emailed on every
  change.

## Stack

- **Next.js 14** (App Router, TypeScript)
- **Tailwind CSS**
- **Prisma + PostgreSQL** for booking storage
- **Resend** for transactional email
- **date-fns**, **zod**

## Setup

```bash
cp .env.example .env.local        # fill in everything
npm install                       # generates the Prisma client
npm run db:push                   # creates the Booking table in Postgres
npm run dev
```

Open http://localhost:3000. The admin sign-in is at `/admin`.

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Postgres connection string (Vercel Postgres, Supabase, etc.) |
| `RESEND_API_KEY` | API key from [resend.com](https://resend.com) |
| `BOOKING_FROM_EMAIL` | Verified sender address |
| `BOOKING_TO_EMAILS` | Comma-separated admin recipients |
| `ADMIN_PASSWORD` | Password for `/admin` login |
| `ADMIN_SESSION_SECRET` | Random string, signs the admin session cookie |
| `BOOKING_ACTION_SECRET` | Random string, signs the email action links |
| `APP_BASE_URL` | Absolute URL of the deployed site (used in email links) |

## Booking flow

1. Visitor submits the form. A row is created in `Booking` with
   `status=PENDING`.
2. The owners receive an email with Accept / Tentative / Decline buttons.
   Each button is a signed, 14-day URL that goes to
   `/api/booking-action?t=…`.
3. The guest also receives a confirmation that their request was received.
4. When an owner clicks an action, the booking status flips
   (`CONFIRMED` / `TENTATIVE` / `DECLINED`), and a status email is sent to
   the guest.
5. The owners can also manage everything from `/admin`: move dates, change
   status, edit notes, cancel, or delete. Every status- or date-change
   triggers an automatic guest email.

Only `CONFIRMED` and `TENTATIVE` bookings block the public availability
calendar.

## Deploy

Push to GitHub and import the project into Vercel. Add a Vercel Postgres
database from the Storage tab, then set all env vars in Project Settings.
On first deploy, run `npm run db:push` from the Vercel CLI or rely on the
included migrations (run `npm run db:migrate -- --name init` locally first
to generate them).
