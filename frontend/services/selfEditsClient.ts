import { VortexApiError } from './vortexClient';

export type SelfEditProposalStatus = 'pending' | 'accepted' | 'rejected' | 'applied' | 'failed';

export type SelfEditFileChange = { path: string; diff: string };

export interface SelfEditProposalSummary {
  id: string;
  created_at: number | null;
  title: string;
  summary: string;
  status: SelfEditProposalStatus;
  author: string;
  files: string[];
}

export interface SelfEditProposalDetail extends SelfEditProposalSummary {
  diff: string;
  fileChanges: SelfEditFileChange[];
  sandbox?: any;
  apply?: any;
}

const joinUrl = (baseUrl: string, path: string) => {
  const base = (baseUrl || '').trim();
  if (!base) return path;
  return base.replace(/\/+$/, '') + path;
};

const buildHeaders = (token?: string) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const trimmed = (token || '').trim();
  if (trimmed) headers['Authorization'] = `Bearer ${trimmed}`;
  return headers;
};

const readErrorFromResponse = async (res: Response) => {
  let message = `HTTP ${res.status}`;
  try {
    const data = await res.json();
    const err = data?.error;
    if (typeof err?.message === 'string') message = err.message;
  } catch {
    try {
      const text = await res.text();
      if (text) message = text;
    } catch {}
  }
  throw new VortexApiError(message, res.status);
};

export const listSelfEditProposals = async (opts: {
  baseUrl: string;
  token?: string;
  status?: SelfEditProposalStatus | 'all';
  signal?: AbortSignal;
}): Promise<SelfEditProposalSummary[]> => {
  const statusParam = (opts.status || 'pending').trim();
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals?status=${encodeURIComponent(statusParam)}`), {
    method: 'GET',
    headers: buildHeaders(opts.token),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  const data = await res.json();
  return Array.isArray(data?.data) ? (data.data as SelfEditProposalSummary[]) : [];
};

export const getSelfEditProposal = async (opts: {
  baseUrl: string;
  token?: string;
  id: string;
  signal?: AbortSignal;
}): Promise<SelfEditProposalDetail> => {
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals/${encodeURIComponent(opts.id)}`), {
    method: 'GET',
    headers: buildHeaders(opts.token),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  return (await res.json()) as SelfEditProposalDetail;
};

export const acceptSelfEditProposal = async (opts: { baseUrl: string; token?: string; id: string; signal?: AbortSignal }) => {
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals/${encodeURIComponent(opts.id)}/accept`), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({}),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  return await res.json();
};

export const rejectSelfEditProposal = async (opts: { baseUrl: string; token?: string; id: string; signal?: AbortSignal }) => {
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals/${encodeURIComponent(opts.id)}/reject`), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({}),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  return await res.json();
};

export const applySelfEditProposal = async (opts: { baseUrl: string; token?: string; id: string; signal?: AbortSignal }) => {
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals/${encodeURIComponent(opts.id)}/apply`), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({}),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  return await res.json();
};

export const createDemoSelfEditProposal = async (opts: { baseUrl: string; token?: string; signal?: AbortSignal }) => {
  const res = await fetch(joinUrl(opts.baseUrl, `/v1/self-edits/proposals/demo`), {
    method: 'POST',
    headers: buildHeaders(opts.token),
    body: JSON.stringify({}),
    signal: opts.signal,
  });
  if (!res.ok) await readErrorFromResponse(res);
  return await res.json();
};

