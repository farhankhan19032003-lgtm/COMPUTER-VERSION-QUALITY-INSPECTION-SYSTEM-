# Sign-in setup: email, Google and GitHub

The dashboard has a full sign-in / create-account screen (email + password, "Continue with Google", "Continue with
GitHub", email confirmation, forgot / reset password). Email works as soon as Supabase is configured. Google and GitHub
need a one-time setup in their own consoles, because only you can register an app under your account.

Until a provider is switched on, its button shows **Not set up** and explains why when clicked. Once you enable it in
Supabase the tag disappears on the next page load; no code change is needed.

Replace `<project-ref>` below with the part of your Supabase URL before `.supabase.co`.

## 1. URL configuration (do this first, it is required)

Supabase dashboard → **Authentication → URL Configuration**

| Field | Local development | Production |
|---|---|---|
| Site URL | `http://localhost:8001` | `https://your-domain.com` |
| Redirect URLs (add) | `http://localhost:8001/**` | `https://your-domain.com/**` |

Without this, the link in the confirmation email and the return trip from Google / GitHub send people to the wrong address
(Supabase falls back to `http://localhost:3000`).

## 2. Email + password

**Authentication → Providers → Email** is on by default.

- **Confirm email** (recommended in production): new users must click the link in an email before they can sign in. The
  dashboard shows a "Check your email" screen with a resend button.
- For quick local testing you can switch **Confirm email** off; sign-up then logs the user in straight away.
- Supabase's built-in mail sender is for testing only and allows just a few emails per hour. For real users set up your
  own SMTP under **Authentication → Emails → SMTP Settings**, and edit the email wording under **Email Templates**.

## 3. Google

1. Open [Google Cloud Console](https://console.cloud.google.com/), create or pick a project.
2. **APIs & Services → OAuth consent screen**: user type *External*, fill in app name and support email, add the scopes
   `openid`, `.../auth/userinfo.email`, `.../auth/userinfo.profile`. While the app is in *Testing*, add your own Google
   account under **Test users**.
3. **APIs & Services → Credentials → Create credentials → OAuth client ID → Web application**.
   Under **Authorized redirect URIs** add exactly:
   `https://<project-ref>.supabase.co/auth/v1/callback`
4. Copy the **Client ID** and **Client secret**.
5. Supabase → **Authentication → Sign In / Providers → Google**: switch on, paste both values, **Save**.

## 4. GitHub

1. GitHub → **Settings → Developer settings → OAuth Apps → New OAuth App**.
2. Homepage URL: your Site URL from step 1.
   Authorization callback URL: `https://<project-ref>.supabase.co/auth/v1/callback`
3. Register, then **Generate a new client secret**. Copy the Client ID and the secret.
4. Supabase → **Authentication → Sign In / Providers → GitHub**: switch on, paste both values, **Save**.

## 5. Phone number (SMS code)

The button **Continue with phone number** texts a 6-digit code and signs the person in; a number that has never been used
gets an account created after the code is confirmed. Number format is international: `+91 98765 43210`.

1. Supabase → **Authentication → Sign In / Providers → Phone**: switch on, pick an SMS provider (Twilio, MessageBird,
   Vonage or TextLocal) and paste its credentials.
2. A **Twilio trial** account can only text numbers you have verified in Twilio, and it adds a "Sent from a Twilio trial
   account" prefix. Anyone else gets "couldn't send the text message" in the dashboard. Upgrade the account to text anyone.
3. SMS costs money per message, and some countries (India, for example) have extra sender-registration rules; check your
   SMS provider's page for the destination country before launch.
4. Supabase limits how often one number can request a code. The dashboard also adds a 30-second wait before "resend".

## 6. Check it

Reload the dashboard signed out. The "Not set up" tag should be gone from the buttons you enabled. Click one, approve,
and you land back on the dashboard with your Google / GitHub name and photo in the top-right menu.

## Good to know

- The Client secrets live only in the Supabase dashboard. Never put them in `.env`, `.env.example` or the HTML.
- By default Supabase links sign-ins that share the same verified email, so someone who registered with email + password
  and later uses Google with the same address ends up in one account.
- The backend needs no change for social login: tokens from every method are verified the same way (see `app/auth.py`).
- If a sign-in error appears in the address bar after you return (for example an expired confirmation link), the screen
  shows it in plain words and offers to try again.
