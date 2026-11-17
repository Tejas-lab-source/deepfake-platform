import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { api, ApiError, setSession } from '../api/client';
import App from '../App';
import FilterLab from '../pages/FilterLab';
import History from '../pages/History';
import ModelCard from '../pages/ModelCard';
import Results from '../pages/Results';
import type { Comparison, ModelCard as Card } from '../api/types';
import { listItem } from './fixtures';

const at = (ui: JSX.Element, path = '/') =>
  render(<MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter>);

describe('History', () => {
  it('shows an empty state with a way to start', async () => {
    vi.spyOn(api, 'listAnalyses').mockResolvedValue({ page: 1, limit: 10, total: 0, items: [] });
    at(<History />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading');
    expect(await screen.findByText('No analyses yet')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Upload an image or video' })).toHaveAttribute(
      'href',
      '/',
    );
  });

  it('pages through results and deletes one after confirmation', async () => {
    const list = vi.spyOn(api, 'listAnalyses').mockImplementation(async (page) => ({
      page,
      limit: 10,
      total: 12,
      items: Array.from({ length: page === 1 ? 10 : 2 }, (_, i) => listItem(page * 100 + i)),
    }));
    const del = vi.spyOn(api, 'deleteAnalysis').mockResolvedValue();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    at(<History />);
    expect(await screen.findByText('Page 1 of 2 · 12 total')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: 'Next' }));
    expect(await screen.findByText('Page 2 of 2 · 12 total')).toBeInTheDocument();
    expect(list).toHaveBeenLastCalledWith(2, 10);
    await userEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]!);
    expect(del).toHaveBeenCalledWith('id200');
    expect(screen.getAllByText(/No manipulation found · 12%/).length).toBeGreaterThan(0);
  });

  it('shows errors with a retry', async () => {
    const list = vi
      .spyOn(api, 'listAnalyses')
      .mockRejectedValueOnce(new ApiError(503, 'Service unavailable'));
    list.mockResolvedValue({ page: 1, limit: 10, total: 1, items: [listItem(1)] });
    at(<History />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Service unavailable');
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByText('Page 1 of 1 · 1 total')).toBeInTheDocument();
  });
});

const card: Card = {
  mockMode: false,
  detectors: {
    primary: 'yermandy/GenD_CLIP_L_14',
    secondary: 'SBI EfficientNet-B4 (FF++ c23)',
    sbiLoaded: true,
    headLoaded: false,
    device: 'cpu',
  },
  pipeline: {
    beautifiedCopiesPerImage: 4,
    beautifiedCopiesPerVideoFrame: 0,
    maxVideoFrames: 32,
    videoTopK: 5,
  },
  calibration: {
    version: 'mc-1',
    status: 'ok',
    fixture: false,
    temperature: 1.2,
    threshold: 0.47,
    uncertainMargin: 0.1,
    confidentMargin: 0.25,
    spreadLimit: 1.5,
    detectorDisagreeLimit: 0.6,
    sources: { threshold: 'fitted on validation split' },
  },
  secondCheck: { provider: 'Hive', mode: 'mock', threshold: 0.9, rule: 'never flips' },
  explanations: { provider: 'Groq', mode: 'off', rule: 'never sees media' },
  licences: [{ component: 'SBI', licence: 'Research use only' }],
  notice: 'Automated, probabilistic detection. A result is not proof of manipulation.',
};

describe('Model card', () => {
  it('shows values from the backend with their sources and licences', async () => {
    vi.spyOn(api, 'modelCard').mockResolvedValue(card);
    at(<ModelCard />);
    expect(await screen.findByText('Calibration (mc-1)')).toBeInTheDocument();
    expect(screen.getByText('0.47')).toBeInTheDocument();
    expect(screen.getByText('fitted on validation split')).toBeInTheDocument();
    expect(screen.getByText('Research use only')).toBeInTheDocument();
    expect(screen.getByText('not loaded (GenD logit)')).toBeInTheDocument();
  });

  it('shows an error when the ML service is down', async () => {
    vi.spyOn(api, 'modelCard').mockRejectedValue(
      new ApiError(503, 'The ML service is not reachable right now.'),
    );
    at(<ModelCard />);
    expect(await screen.findByRole('alert')).toHaveTextContent('not reachable');
  });
});

describe('Results', () => {
  it('says nothing has been imported rather than showing numbers', async () => {
    vi.spyOn(api, 'evalRuns').mockResolvedValue({ report: null });
    at(<Results />);
    expect(await screen.findByText('No evaluation results yet')).toBeInTheDocument();
  });

  it('flags fixture reports and tabulates runs', async () => {
    vi.spyOn(api, 'evalRuns').mockResolvedValue({
      report: {
        id: 'r',
        created: '2026-11-17',
        fixture: true,
        notice: 'PLUMBING FIXTURE',
        aggregate: { 'H-a': { clean_auc: { mean: 0.81 } }, 'H-f': { clean_auc: { mean: 0.8 } } },
        successCriteria: [{ name: 'flip_rate_reduction', met: true }],
        runs: [
          {
            scheme: 'H-a',
            fold: 'F1',
            seed: 0,
            heldOut: true,
            metrics: { clean_auc: 0.81 },
            ci95: { clean_auc: [0.7, 0.9] },
          },
        ],
      },
    });
    at(<Results />);
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'PLUMBING FIXTURE: these numbers test the plumbing',
    );
    expect(screen.getAllByText('0.810').length).toBeGreaterThan(0);
    expect(screen.getByTitle('95% CI 0.700–0.900')).toBeInTheDocument();
    expect(screen.getByText('met')).toBeInTheDocument();
  });
});

const comparison = (flipped: boolean): Comparison => ({
  id: 'c1',
  family: 'F1',
  strength: 0.6,
  threshold: 0.5,
  clean: {
    verdict: 'FAKE',
    probability: 0.62,
    confident: false,
    uncertainReason: null,
    scores: null,
    image: null,
    heatmap: null,
  },
  filtered: {
    verdict: flipped ? 'REAL' : 'FAKE',
    probability: 0.41,
    confident: false,
    uncertainReason: null,
    scores: null,
    image: null,
    heatmap: null,
  },
  flipped,
  heatmapIoU: 0.37,
  createdAt: '2026-11-17T10:00:00Z',
});

describe('Filter Lab', () => {
  it('highlights a changed verdict', async () => {
    vi.spyOn(api, 'listAnalyses').mockResolvedValue({
      page: 1,
      limit: 50,
      total: 1,
      items: [listItem(1)],
    });
    vi.spyOn(api, 'comparisons').mockResolvedValue({ items: [] });
    const compare = vi.spyOn(api, 'compare').mockResolvedValue(comparison(true));
    at(<FilterLab />, '/filter-lab?analysis=id1');
    await userEvent.click(await screen.findByRole('button', { name: 'Compare' }));
    expect(compare).toHaveBeenCalledWith('id1', 'F1', 0.6);
    expect(
      await screen.findByText(/The verdict changed under this filter \(F1 at 60%\)/),
    ).toBeInTheDocument();
    expect(screen.getByText('Heatmap overlap (IoU of the top regions):')).toBeInTheDocument();
  });

  it('explains clearly when the real pipeline is not available, inventing nothing', async () => {
    vi.spyOn(api, 'listAnalyses').mockResolvedValue({
      page: 1,
      limit: 50,
      total: 1,
      items: [listItem(1)],
    });
    vi.spyOn(api, 'comparisons').mockResolvedValue({ items: [] });
    vi.spyOn(api, 'compare').mockRejectedValue(
      new ApiError(501, 'Filters need the real pipeline; set MOCK_MODE=false.'),
    );
    at(<FilterLab />, '/filter-lab?analysis=id1');
    await userEvent.click(await screen.findByRole('button', { name: 'Compare' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The Filter Lab needs the real ML pipeline',
    );
    expect(screen.queryByText(/The verdict (changed|stayed)/)).not.toBeInTheDocument();
  });

  it('asks for an analysed image when there is none', async () => {
    vi.spyOn(api, 'listAnalyses').mockResolvedValue({ page: 1, limit: 50, total: 0, items: [] });
    at(<FilterLab />, '/filter-lab');
    expect(await screen.findByText('No analysed images yet')).toBeInTheDocument();
  });
});

describe('App shell', () => {
  it('shows working navigation when signed in, and the login page otherwise', async () => {
    vi.spyOn(api, 'health').mockResolvedValue({ status: 'ok', mockMode: true });
    vi.spyOn(api, 'evalRuns').mockResolvedValue({ report: null });
    setSession({
      user: { id: 'u', email: 'u@example.test', role: 'user' },
      accessToken: 'a',
      refreshToken: 'r',
    });
    at(<App />, '/results');
    const nav = screen.getByRole('navigation', { name: 'Main' });
    for (const name of ['Detect', 'History', 'Filter Lab', 'Results', 'Model card']) {
      expect(screen.getByRole('link', { name })).toBeInTheDocument();
    }
    expect(nav).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Results' })).toHaveAttribute('aria-current', 'page');
    expect(await screen.findByText(/Mock mode: scores are placeholders/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Log out' }));
    expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument();
  });
});
