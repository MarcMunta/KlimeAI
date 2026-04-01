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
      <div className="absolute inset-[-10%] rounded-full bg-primary/18 blur-xl opacity-70" />
      <img
        src="/vortex-logo.svg"
        alt={alt}
        draggable={false}
        className="relative h-full w-full object-contain drop-shadow-[0_16px_34px_rgba(0,194,255,0.24)]"
      />
    </div>
  );
};

export default VortexLogo;
