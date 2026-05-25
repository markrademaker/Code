// Hardcoded baseline so the site works out of the box. Override at
// runtime by setting ADMIN_EMAILS to a comma-separated list.
const HARDCODED_ADMIN_EMAILS = [
  "rademair@gmail.com",
  "neleman3@gmail.com",
  "mtrademaker@gmail.com",
];

function getAdminEmails(): string[] {
  const env = process.env.ADMIN_EMAILS;
  if (env && env.trim()) {
    return env
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter(Boolean);
  }
  return HARDCODED_ADMIN_EMAILS.map((e) => e.toLowerCase());
}

export function isAdminEmail(email: string | null | undefined): boolean {
  if (!email) return false;
  return getAdminEmails().includes(email.toLowerCase().trim());
}
