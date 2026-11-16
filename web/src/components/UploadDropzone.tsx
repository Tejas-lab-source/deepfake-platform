import { useRef, useState, type DragEvent } from 'react';

interface Props {
  disabled: boolean;
  onFile: (file: File) => void;
}

const ACCEPT = 'image/jpeg,image/png,image/webp,video/mp4,video/webm,video/quicktime';

export default function UploadDropzone({ disabled, onFile }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function drop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    const file = e.dataTransfer.files[0];
    if (file && !disabled) onFile(file);
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
      className={`w-full rounded-lg border-2 border-dashed px-6 py-10 text-center transition disabled:opacity-50 ${
        over ? 'border-slate-900 bg-slate-100' : 'border-slate-300 bg-white hover:border-slate-500'
      }`}
    >
      <span className="block font-medium">Drop a face image or short video here</span>
      <span className="mt-1 block text-sm text-slate-500">
        or click to choose. JPEG, PNG or WebP up to 10 MB; MP4, WebM or MOV up to 100 MB.
      </span>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = '';
        }}
      />
    </button>
  );
}
