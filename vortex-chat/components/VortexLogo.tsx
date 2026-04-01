import React from 'react';

interface VortexLogoProps {
  alt?: string;
  className?: string;
  size?: number;
}

const VortexLogo: React.FC<VortexLogoProps> = ({
  alt = 'Vortex logo',
  className = '',
  size = 56,
}) => {
  return (
    <div
      className={`relative shrink-0 select-none ${className}`}
      style={{ width: size, height: size }}
    >
      <div className="absolute inset-[-8%] rounded-full bg-primary/10 blur-md opacity-75" />
      <img
        src="/vortex-logo.svg"
        alt={alt}
        draggable={false}
        className="relative h-full w-full object-contain drop-shadow-[0_10px_24px_rgba(16,163,127,0.18)]"
      />
    </div>
  );
};

export default VortexLogo;
