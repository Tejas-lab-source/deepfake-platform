import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '../api/client';
import AskPanel from '../components/AskPanel';
import FrameTimeline from '../components/FrameTimeline';
import ProgressSteps from '../components/ProgressSteps';
import ResultView from '../components/ResultView';
import { frames, imageAnalysis, videoAnalysis } from './fixtures';

const inRouter = (ui: JSX.Element) => render(<MemoryRouter>{ui}</MemoryRouter>);
const stubImages = () => vi.spyOn(api, 'image').mockResolvedValue(new Blob(['png']));

describe('FrameTimeline', () => {
  it('draws faceless and undecoded frames as gaps, marks extremes and the threshold', () => {
    const ml = videoAnalysis().ml!;
    render(<FrameTimeline analysisId="v1" ml={ml} threshold={0.5} />);
    expect(screen.getAllByTestId('frame-bar')).toHaveLength(2);
    expect(screen.getAllByTestId('frame-gap')).toHaveLength(2);
    const list = screen.getByRole('list', { name: 'Frame timeline' });
    expect(
      within(list).getByLabelText(/Frame 15 at 1\.50 s: no face detected \(excluded\)/),
    ).toBeInTheDocument();
    expect(
      within(list).getByLabelText(/Frame 35 at 3\.50 s: could not be decoded/),
    ).toBeInTheDocument();
    expect(
      within(list).getByLabelText(
        /Frame 25 at 2\.50 s: score 92%, above threshold, most suspicious/,
      ),
    ).toBeInTheDocument();
    expect(
      within(list).getByLabelText(
        /Frame 5 at 0\.50 s: score 67%, above threshold, least suspicious/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /4 frames sampled, 2 with a scored face, 4\.0 s of video; verdict from the top 2 frames/,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText('Heatmaps were made for frames 5 and 25.')).toBeInTheDocument();
  });

  it("shows a frame's details, including why a gap does not count", async () => {
    const ml = videoAnalysis().ml!;
    render(<FrameTimeline analysisId="v1" ml={ml} threshold={0.5} />);
    await userEvent.click(screen.getByLabelText(/Frame 15 at/));
    expect(screen.getByText(/does not count towards the verdict/)).toBeInTheDocument();
    await userEvent.click(screen.getByText(/Most suspicious: frame 25 at 2\.50 s/));
    expect(screen.getByText(/Head logit/)).toHaveTextContent(
      'Head logit 2.40, face probability 92%.',
    );
  });

  it('handles a single frame and a result without frames', () => {
    const one = { frames: [frames[0]!], framesSampled: 1, framesScored: 1 };
    const { unmount } = render(<FrameTimeline analysisId="v" ml={one} threshold={0.5} />);
    expect(screen.getAllByTestId('frame-bar')).toHaveLength(1);
    unmount();
    render(<FrameTimeline analysisId="v" ml={{ frames: [] }} threshold={0.5} />);
    expect(screen.getByText('No frame results are available for this video.')).toBeInTheDocument();
  });
});

describe('ProgressSteps', () => {
  it('shows measured progress and upload percentage; images skip frame extraction', () => {
    const { rerender } = render(
      <ProgressSteps
        stage="scoring"
        progress={{ done: 12, total: 32, unit: 'frames' }}
        kind="video"
      />,
    );
    expect(screen.getByText('12 of 32 frames')).toBeInTheDocument();
    const steps = screen.getByRole('list', { name: 'Progress' });
    expect(within(steps).getByText('Running detectors')).toHaveAttribute('aria-current', 'step');
    expect(screen.getByRole('progressbar', { name: 'Running detectors' })).toHaveAttribute(
      'aria-valuenow',
      '38',
    );
    expect(screen.getByText('Extracting frames')).toBeInTheDocument();
    rerender(<ProgressSteps stage="queued" kind="image" upload={0.4} />);
    expect(screen.queryByText('Extracting frames')).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Uploading' })).toHaveAttribute(
      'aria-valuenow',
      '40',
    );
  });
});

describe('ResultView', () => {
  it('shows the verdict, checks, explanation, heatmaps, Ask and downloads, with the disclaimer', async () => {
    stubImages();
    inRouter(<ResultView analysis={imageAnalysis()} />);
    expect(screen.getByRole('heading', { name: 'Likely manipulated' })).toBeInTheDocument();
    expect(screen.getByText('Single check only')).toBeInTheDocument();
    expect(screen.getByRole('meter', { name: 'Likelihood of manipulation' })).toHaveAttribute(
      'aria-valuenow',
      '91',
    );
    expect(
      screen.getByText(/Only our pipeline decided \(the second check was switched off/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Switched off for this upload/)).toBeInTheDocument();
    expect(
      screen.getByText('The automated check found signs of manipulation.'),
    ).toBeInTheDocument();
    expect(screen.getByLabelText('Heatmap opacity')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download PDF report' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download JSON' })).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ask about this result…')).toBeInTheDocument();
    expect(screen.getAllByText(/not proof/).length).toBeGreaterThan(1);
    expect(await screen.findAllByRole('img')).not.toHaveLength(0);
  });

  it('gives UNCERTAIN its own explanation and shows no score when there is none', () => {
    const a = imageAnalysis({
      final: { verdict: 'UNCERTAIN', badge: 'single_check', ruleApplied: 'ours_uncertain' },
      ours: {
        verdict: 'UNCERTAIN',
        probability: null,
        threshold: 0.5,
        confident: false,
        uncertainReason: 'no_face_detected',
        spread: null,
      },
      ml: { faces: [] },
    });
    inRouter(<ResultView analysis={a} />);
    expect(screen.getByRole('heading', { name: 'Uncertain' })).toBeInTheDocument();
    expect(
      screen.getByText(/No reliable verdict: No face detected \(never counted as real\)/),
    ).toBeInTheDocument();
    expect(screen.getByText('No score: nothing could be scored.')).toBeInTheDocument();
    expect(screen.queryByRole('meter')).not.toBeInTheDocument(); // nothing invented
    expect(screen.getByText('No face was detected.')).toBeInTheDocument();
  });

  it('renders the frame timeline for videos and downloads the PDF', async () => {
    const report = vi.spyOn(api, 'report').mockResolvedValue(new Blob(['%PDF']));
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined);
    inRouter(<ResultView analysis={videoAnalysis()} />);
    expect(screen.getByRole('list', { name: 'Frame timeline' })).toBeInTheDocument();
    expect(screen.getByText(/2 of 4 sampled frames had a scored face/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Download PDF report' }));
    expect(report).toHaveBeenCalledWith('v1');
    expect(click).toHaveBeenCalled();
  });

  it('reports a PDF that could not be generated', async () => {
    vi.spyOn(api, 'report').mockRejectedValue(new ApiError(409, 'Analysis not finished'));
    inRouter(<ResultView analysis={imageAnalysis({ ml: { faces: [] } })} />);
    await userEvent.click(screen.getByRole('button', { name: 'Download PDF report' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The report could not be generated.',
    );
  });

  it('labels mock results as placeholders', () => {
    inRouter(<ResultView analysis={imageAnalysis({ ml: { mock: true, faces: [] } })} />);
    expect(screen.getByText(/Mock mode: the scores are placeholders/)).toBeInTheDocument();
  });
});

describe('AskPanel', () => {
  it('sends a question and shows the answer', async () => {
    const ask = vi.spyOn(api, 'ask').mockResolvedValue({
      answer: 'Final result: FAKE. This is an automated check and is not proof either way.',
      source: 'template',
      intent: 'verdict',
      fallbackReason: null,
    });
    render(<AskPanel analysisId="a1" />);
    await userEvent.type(screen.getByLabelText('Question about this result'), 'Is it fake?');
    await userEvent.click(screen.getByRole('button', { name: 'Ask' }));
    expect(ask).toHaveBeenCalledWith('a1', 'Is it fake?');
    expect(await screen.findByText(/Final result: FAKE/)).toBeInTheDocument();
  });

  it('offers suggestions and shows errors such as a rate limit', async () => {
    vi.spyOn(api, 'ask').mockRejectedValue(
      new ApiError(429, 'Rate limit reached; try again in 60 s.'),
    );
    render(<AskPanel analysisId="a1" />);
    await userEvent.click(screen.getByRole('button', { name: 'Why was this flagged?' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Rate limit reached');
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled(); // empty input
  });
});
