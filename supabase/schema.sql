-- VisionInspect schema. Run in Supabase: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------- inspections
create table if not exists public.inspections (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid references auth.users(id) on delete set null,
    source          text not null default 'upload',          -- upload | stream | camera
    product_type    text not null default 'unknown',         -- phone | laptop | watch | ...
    verdict         text not null check (verdict in ('PASS','REVIEW','FAIL')),
    score           double precision not null default 0,     -- 0 (clean) .. 1 (defective)
    anomaly_ratio   double precision,                        -- score / calibrated threshold
    defects         jsonb not null default '[]'::jsonb,
    detections      jsonb not null default '[]'::jsonb,
    quality         jsonb not null default '{}'::jsonb,
    image_path      text,
    annotated_path  text,
    latency_ms      double precision not null default 0,
    created_at      timestamp with time zone not null default now()
);
create index if not exists inspections_user_created_idx on public.inspections (user_id, created_at desc);
create index if not exists inspections_verdict_idx on public.inspections (verdict);
create index if not exists inspections_product_idx on public.inspections (product_type);

-- ------------------------------------------------------------ reviewer feedback
-- A human confirms or overrides the model's verdict. These rows are the labelled examples used to
-- measure real accuracy (agreement, false alarms, misses) and to retrain.
create table if not exists public.inspection_feedback (
    id                uuid primary key default gen_random_uuid(),
    inspection_id     uuid not null references public.inspections(id) on delete cascade,
    user_id           uuid references auth.users(id) on delete set null,   -- the reviewer
    model_verdict     text not null check (model_verdict in ('PASS','REVIEW','FAIL')),
    reviewer_verdict  text not null check (reviewer_verdict in ('PASS','REVIEW','FAIL')),
    note              text,
    created_at        timestamp with time zone not null default now(),
    updated_at        timestamp with time zone not null default now()
);
create unique index if not exists feedback_inspection_user_idx on public.inspection_feedback (inspection_id, user_id);
create index if not exists feedback_user_created_idx on public.inspection_feedback (user_id, created_at desc);

-- ------------------------------------------------------------------- patients
-- PERSONAL HEALTH DATA. Keep it minimal: name and ids are optional, store only what you need.
-- consent_given must be true and consent_at is recorded; deleting a patient cascades to their
-- screenings (right to erasure). Images in the storage bucket must be removed by your retention job.
create table if not exists public.patients (
    id               uuid primary key default gen_random_uuid(),
    user_id          uuid not null references auth.users(id) on delete cascade,   -- the clinician / operator
    full_name        text,
    external_ref     text,                                   -- hospital / clinic id, optional
    age_years        integer check (age_years between 0 and 130),
    sex              text check (sex in ('female','male','other','unknown')) default 'unknown',
    skin_type        text check (skin_type in ('I','II','III','IV','V','VI','unknown')) default 'unknown',  -- Fitzpatrick
    consent_given    boolean not null check (consent_given),
    consent_at       timestamp with time zone not null default now(),
    consent_version  text not null default 'v1',
    notes            text,
    created_at       timestamp with time zone not null default now()
);
create index if not exists patients_user_created_idx on public.patients (user_id, created_at desc);

-- ----------------------------------------------------------------- screenings
create table if not exists public.screenings (
    id             uuid primary key default gen_random_uuid(),
    user_id        uuid references auth.users(id) on delete set null,
    module         text not null,                             -- skin_lesion | chest_xray | ...
    top_label      text not null,
    probabilities  jsonb not null default '{}'::jsonb,
    triage         text not null check (triage in ('LOW','MODERATE','HIGH','INCONCLUSIVE')),
    quality        jsonb not null default '{}'::jsonb,
    image_path     text,
    latency_ms     double precision not null default 0,
    patient_id     uuid references public.patients(id) on delete cascade,
    created_at     timestamp with time zone not null default now()
);
-- for databases created before patients existed (safe to re-run):
alter table public.screenings add column if not exists patient_id uuid references public.patients(id) on delete cascade;
create index if not exists screenings_user_created_idx on public.screenings (user_id, created_at desc);
create index if not exists screenings_patient_idx on public.screenings (patient_id);

-- ------------------------------------------------------------------ audit log
-- Who did what to which patient record. Holds NO personal data in meta (ids and counts only).
create table if not exists public.audit_log (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid,
    action      text not null,                                 -- patient.create | patient.read | patient.delete | screening.create | report.create
    patient_id  uuid,                                          -- no FK: the record must outlive a deleted patient
    meta        jsonb not null default '{}'::jsonb,
    created_at  timestamp with time zone not null default now()
);
create index if not exists audit_user_created_idx on public.audit_log (user_id, created_at desc);

-- ---------------------------------------------------------- phone diagnostics
create table if not exists public.phone_diagnostics (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid references auth.users(id) on delete set null,
    platform      text not null default 'android',
    model         text,
    health_score  integer not null,
    verdict       text not null,
    findings      jsonb not null default '[]'::jsonb,
    input         jsonb not null default '{}'::jsonb,
    created_at    timestamp with time zone not null default now()
);
create index if not exists phone_diag_user_created_idx on public.phone_diagnostics (user_id, created_at desc);

-- -------------------------------------------------------------- system events
create table if not exists public.system_events (
    id          uuid primary key default gen_random_uuid(),
    level       text not null check (level in ('INFO','WARNING','ERROR')),
    message     text not null,
    meta        jsonb not null default '{}'::jsonb,
    created_at  timestamp with time zone not null default now()
);
create index if not exists system_events_created_idx on public.system_events (created_at desc);

-- ------------------------------------------------------------ row level security
alter table public.inspections   enable row level security;
alter table public.screenings    enable row level security;
alter table public.system_events enable row level security;
alter table public.phone_diagnostics enable row level security;
alter table public.patients enable row level security;
alter table public.audit_log enable row level security;
alter table public.inspection_feedback enable row level security;

-- Users only see and create their own rows. The backend uses the service role
-- key, which bypasses RLS, and always sets user_id from the verified token.
drop policy if exists "inspections_select_own" on public.inspections;
create policy "inspections_select_own" on public.inspections
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists "inspections_insert_own" on public.inspections;
create policy "inspections_insert_own" on public.inspections
    for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "screenings_select_own" on public.screenings;
create policy "screenings_select_own" on public.screenings
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists "screenings_insert_own" on public.screenings;
create policy "screenings_insert_own" on public.screenings
    for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "phone_diag_select_own" on public.phone_diagnostics;
create policy "phone_diag_select_own" on public.phone_diagnostics
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists "phone_diag_insert_own" on public.phone_diagnostics;
create policy "phone_diag_insert_own" on public.phone_diagnostics
    for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "feedback_select_own" on public.inspection_feedback;
create policy "feedback_select_own" on public.inspection_feedback
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists "feedback_insert_own" on public.inspection_feedback;
create policy "feedback_insert_own" on public.inspection_feedback
    for insert to authenticated with check (auth.uid() = user_id);

drop policy if exists "patients_select_own" on public.patients;
create policy "patients_select_own" on public.patients
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists "patients_insert_own" on public.patients;
create policy "patients_insert_own" on public.patients
    for insert to authenticated with check (auth.uid() = user_id and consent_given);

-- system_events and audit_log: no policies for end users = service role only (users cannot edit history).

-- --------------------------------------------------------------------- storage
-- Private bucket. Files are stored under "<user_id>/<yyyy-mm-dd>/<uuid>.jpg".
insert into storage.buckets (id, name, public)
values ('inspections', 'inspections', false)
on conflict (id) do nothing;

drop policy if exists "inspections_read_own_files" on storage.objects;
create policy "inspections_read_own_files" on storage.objects
    for select to authenticated
    using (bucket_id = 'inspections' and (storage.foldername(name))[1] = auth.uid()::text);

-- ------------------------------------------------------------------ analytics
create or replace view public.inspection_daily as
select
    user_id,
    date_trunc('day', created_at)::date                      as day,
    count(*)                                                 as total,
    count(*) filter (where verdict = 'PASS')                 as passed,
    count(*) filter (where verdict = 'REVIEW')               as review,
    count(*) filter (where verdict = 'FAIL')                 as failed,
    round(avg(latency_ms)::numeric, 1)                       as avg_latency_ms
from public.inspections
group by user_id, date_trunc('day', created_at);

alter view public.inspection_daily set (security_invoker = true);
