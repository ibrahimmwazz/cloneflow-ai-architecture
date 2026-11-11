-- CloneFlow AI - Supabase Database Schema
-- Run this in your Supabase SQL editor

-- ===========================================
-- Enable required extensions
-- ===========================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================
-- Users table (extends Supabase auth.users)
-- ===========================================
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    stripe_customer_id TEXT,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own profile"
    ON public.users FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.users FOR UPDATE
    USING (auth.uid() = id);

-- ===========================================
-- User Credits table
-- ===========================================
CREATE TABLE IF NOT EXISTS public.user_credits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    credits INTEGER DEFAULT 0 NOT NULL,
    credits_used_this_month INTEGER DEFAULT 0 NOT NULL,
    subscription_tier TEXT DEFAULT 'free',
    stripe_subscription_id TEXT,
    subscription_status TEXT DEFAULT 'active',
    current_period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Enable RLS
ALTER TABLE public.user_credits ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own credits"
    ON public.user_credits FOR SELECT
    USING (auth.uid() = user_id);

-- ===========================================
-- Jobs table
-- ===========================================
CREATE TABLE IF NOT EXISTS public.jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL UNIQUE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    runpod_id TEXT,
    status TEXT DEFAULT 'pending' NOT NULL,
    input_data JSONB NOT NULL,
    video_url TEXT,
    error_message TEXT,
    credits_estimated INTEGER DEFAULT 10,
    credits_used INTEGER DEFAULT 0,
    credits_deducted BOOLEAN DEFAULT false,
    execution_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON public.jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON public.jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON public.jobs(job_id);
CREATE INDEX IF NOT EXISTS idx_jobs_runpod_id ON public.jobs(runpod_id);
-- Composite index for paginated user job list queries (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_jobs_user_status ON public.jobs(user_id, status, created_at DESC);

-- Enable RLS
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own jobs"
    ON public.jobs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create jobs"
    ON public.jobs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own jobs"
    ON public.jobs FOR UPDATE
    USING (auth.uid() = user_id);

-- ===========================================
-- Transactions table
-- ===========================================
CREATE TABLE IF NOT EXISTS public.transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    type TEXT NOT NULL, -- 'purchase', 'usage', 'refund', 'bonus'
    credits INTEGER NOT NULL,
    description TEXT,
    stripe_payment_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON public.transactions(user_id);

-- Enable RLS
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own transactions"
    ON public.transactions FOR SELECT
    USING (auth.uid() = user_id);

-- ===========================================
-- User Voices table (for cloned voices)
-- ===========================================
CREATE TABLE IF NOT EXISTS public.user_voices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    voice_id TEXT NOT NULL,
    voice_name TEXT,
    sample_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, voice_id)
);

-- Enable RLS
ALTER TABLE public.user_voices ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own voices"
    ON public.user_voices FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own voices"
    ON public.user_voices FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- ===========================================
-- Functions
-- ===========================================

-- Function to add credits
CREATE OR REPLACE FUNCTION public.add_credits(
    p_user_id UUID,
    p_credits INTEGER
)
RETURNS void AS $$
BEGIN
    INSERT INTO public.user_credits (user_id, credits)
    VALUES (p_user_id, p_credits)
    ON CONFLICT (user_id)
    DO UPDATE SET
        credits = user_credits.credits + p_credits,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to deduct credits (atomic — uses single conditional UPDATE, no TOCTOU race)
-- The WHERE clause checks balance AND performs deduction atomically in one DB round-trip.
CREATE OR REPLACE FUNCTION public.deduct_credits(
    p_user_id UUID,
    p_credits INTEGER
)
RETURNS boolean AS $$
DECLARE
    rows_updated INTEGER;
BEGIN
    UPDATE public.user_credits
    SET
        credits = credits - p_credits,
        credits_used_this_month = credits_used_this_month + p_credits,
        updated_at = NOW()
    WHERE user_id = p_user_id
      AND credits >= p_credits;  -- Atomic guard: only deducts if balance is sufficient
    
    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RETURN rows_updated > 0;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to handle new user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
    INSERT INTO public.users (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    
    INSERT INTO public.user_credits (user_id, credits)
    VALUES (NEW.id, 10); -- 10 free credits for new users
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create user record on signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ===========================================
-- Storage Buckets
-- ===========================================
-- Create bucket for user uploads
INSERT INTO storage.buckets (id, name, public)
VALUES ('user-uploads', 'user-uploads', false)
ON CONFLICT (id) DO NOTHING;

-- Create bucket for generated videos
INSERT INTO storage.buckets (id, name, public)
VALUES ('generated-videos', 'generated-videos', true)
ON CONFLICT (id) DO NOTHING;

-- Storage policies
CREATE POLICY "Users can upload own files"
    ON storage.objects FOR INSERT
    WITH CHECK (bucket_id = 'user-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view own uploads"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'user-uploads' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Anyone can view generated videos"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'generated-videos');

-- ===========================================
-- Marketplace Items table
-- ===========================================
CREATE TABLE IF NOT EXISTS public.marketplace_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    description TEXT,
    type TEXT NOT NULL, -- 'avatar', 'template', etc.
    price_credits INTEGER DEFAULT 0,
    preview_url TEXT,
    asset_url TEXT,
    author_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.marketplace_items ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Anyone can view marketplace items"
    ON public.marketplace_items FOR SELECT
    USING (true);

-- Only admins can insert (checked in application logic/API)
CREATE POLICY "Admins can insert marketplace items"
    ON public.marketplace_items FOR INSERT
    WITH CHECK (EXISTS (SELECT 1 FROM public.users WHERE id = auth.uid() AND is_admin = true));

-- ===========================================
-- User Purchases table
-- ===========================================
CREATE TABLE IF NOT EXISTS public.user_purchases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    item_id UUID NOT NULL REFERENCES public.marketplace_items(id) ON DELETE CASCADE,
    purchase_price INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, item_id)
);

-- Enable RLS
ALTER TABLE public.user_purchases ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Users can view own purchases"
    ON public.user_purchases FOR SELECT
    USING (auth.uid() = user_id);

-- ===========================================
-- RAG Knowledge Base (pgvector) & AI Governance
-- ===========================================
CREATE TABLE IF NOT EXISTS public.brand_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_type TEXT DEFAULT 'brand_guide',
    content_hash TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, content_hash)
);

CREATE TABLE IF NOT EXISTS public.brand_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES public.brand_documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_estimate INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_brand_chunks_user_doc
    ON public.brand_chunks(user_id, document_id);

CREATE INDEX IF NOT EXISTS idx_brand_chunks_embedding_hnsw
    ON public.brand_chunks USING hnsw (embedding vector_cosine_ops);

ALTER TABLE public.brand_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.brand_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own brand documents"
    ON public.brand_documents FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own brand chunks"
    ON public.brand_chunks FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE OR REPLACE FUNCTION public.match_brand_chunks(
    query_embedding vector(384),
    p_user_id UUID,
    match_threshold FLOAT DEFAULT 0.15,
    match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        bc.id,
        bc.document_id,
        bc.chunk_index,
        bc.content,
        bc.metadata,
        1 - (bc.embedding <=> query_embedding) AS similarity
    FROM public.brand_chunks bc
    WHERE bc.user_id = p_user_id
      AND 1 - (bc.embedding <=> query_embedding) >= match_threshold
    ORDER BY bc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TABLE IF NOT EXISTS public.agent_eval_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
    suite_name TEXT NOT NULL,
    metrics JSONB NOT NULL,
    passed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.agent_eval_runs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own eval runs"
    ON public.agent_eval_runs FOR SELECT
    USING (auth.uid() = user_id);

