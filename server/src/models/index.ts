import mongoose, { Schema, type InferSchemaType } from 'mongoose';

const userSchema = new Schema({
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  passwordHash: { type: String, required: true },
  role: { type: String, enum: ['user', 'admin'], default: 'user' },
  createdAt: { type: Date, default: Date.now },
});

const mediaSchema = new Schema({
  ownerId: { type: Schema.Types.ObjectId, ref: 'User', required: true, index: true },
  kind: { type: String, enum: ['image', 'video'], required: true },
  storageKey: { type: String, required: true },
  mime: { type: String, required: true },
  sha256: { type: String, required: true, index: true },
  sizeBytes: { type: Number, required: true },
  createdAt: { type: Date, default: Date.now },
  // MongoDB removes the document once expiresAt has passed.
  expiresAt: { type: Date, required: true, index: { expireAfterSeconds: 0 } },
});

const analysisSchema = new Schema(
  {
    ownerId: { type: Schema.Types.ObjectId, ref: 'User', required: true, index: true },
    mediaId: { type: Schema.Types.ObjectId, ref: 'Media', required: true },
    kind: { type: String, enum: ['image', 'video'], required: true },
    status: { type: String, enum: ['queued', 'processing', 'done', 'failed'], default: 'queued' },
    stage: { type: String, default: 'queued' },
    error: { type: String, default: null },
    externalCheckRequested: { type: Boolean, default: false },
    final: {
      type: new Schema({ verdict: String, badge: String, ruleApplied: String }, { _id: false }),
      default: null,
    },
    ours: {
      type: new Schema(
        {
          verdict: String,
          probability: Number,
          threshold: Number,
          confident: Boolean,
          uncertainReason: String,
          detectorScores: Schema.Types.Mixed,
          spread: Number,
        },
        { _id: false },
      ),
      default: null,
    },
    externalCheckId: { type: Schema.Types.ObjectId, ref: 'ExternalCheck', default: null },
    faces: { type: [Schema.Types.Mixed], default: [] },
    frames: { type: [Schema.Types.Mixed], default: [] },
    heatmapKeys: { type: [String], default: [] },
    modelCardVersion: { type: String, default: null },
    // Full ML service response, kept for the JSON download and the report.
    ml: { type: Schema.Types.Mixed, default: null },
  },
  { timestamps: true },
);
analysisSchema.index({ ownerId: 1, createdAt: -1 });

const externalCheckSchema = new Schema(
  {
    mediaSha256: { type: String, required: true },
    provider: { type: String, required: true },
    status: { type: String, enum: ['ok', 'error', 'timeout', 'quota'], required: true },
    deepfakeScore: { type: Number, default: null },
    aiGeneratedScore: { type: Number, default: null },
    threshold: { type: Number, required: true },
    verdict: { type: String, enum: ['FAKE', 'REAL', null], default: null },
    latencyMs: { type: Number, default: null },
    rawResponse: { type: Schema.Types.Mixed, default: null },
    // Phase 4: why no scores were produced (not_configured, budget:user_limit, auth, ...).
    reason: { type: String, default: null },
    // Phase 4: per-frame scores (video); deepfakeScore/aiGeneratedScore are their maxima.
    frames: {
      type: [
        new Schema(
          { frameIndex: Number, timestamp: Number, deepfake: Number, aiGenerated: Number },
          { _id: false },
        ),
      ],
      default: [],
    },
  },
  { timestamps: true },
);
externalCheckSchema.index({ mediaSha256: 1, provider: 1, status: 1 });

const explanationSchema = new Schema(
  {
    analysisId: { type: Schema.Types.ObjectId, ref: 'Analysis', required: true, unique: true },
    groqModel: { type: String, default: null },
    promptVersion: { type: String, required: true },
    summary: { type: String, required: true },
    why: { type: String, required: true },
    caveats: { type: [String], default: [] },
    fallbackUsed: { type: Boolean, required: true },
  },
  { timestamps: true },
);

export type UserDoc = InferSchemaType<typeof userSchema>;

export const User = mongoose.model('User', userSchema, 'users');
export const Media = mongoose.model('Media', mediaSchema, 'media');
export const Analysis = mongoose.model('Analysis', analysisSchema, 'analyses');
export const ExternalCheck = mongoose.model('ExternalCheck', externalCheckSchema, 'externalChecks');
export const Explanation = mongoose.model('Explanation', explanationSchema, 'explanations');
