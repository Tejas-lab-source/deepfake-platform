import { sha256File } from '../hash.js';
import type { Verifier, VerifierResult } from './verifier.js';

/**
 * Stand-in for Hive in MOCK_MODE. Scores are derived from the file's sha256, so they are
 * repeatable and carry no information about the image. No network call is made.
 */
export class MockHiveVerifier implements Verifier {
  readonly provider = 'hive-mock';

  async verify(filePath: string): Promise<VerifierResult> {
    const digest = Buffer.from(await sha256File(filePath), 'hex');
    // Bytes 20-23 are not used by the ML mock, so the two mock checks are independent.
    const deepfakeScore = digest.readUInt16BE(20) / 65535;
    const aiGeneratedScore = digest.readUInt16BE(22) / 65535;
    return { status: 'ok', deepfakeScore, aiGeneratedScore, raw: { mock: true } };
  }
}
