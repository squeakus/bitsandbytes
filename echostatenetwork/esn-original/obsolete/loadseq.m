function [OSEQ] = loadseq(FILENAME, CDTYPE)
%
%

[fid, msg] = fopen(FILENAME, 'r');
if fid == -1,
   error(sprintf('Error opening file %s (%s)',FILENAME, msg));
end;

seq = fscanf(fid, '%s');
fclose(fid);

slen = length(seq);
if slen < 1,
   error(sprintf('No symbols in file %s', FILENAME));
end;

[sym, dmy, indseq] = unique(seq);

if nargin < 2, CDTYPE = 'HOT'; end;

if CDTYPE == 'BIN', codes = linspace(0,1,length(sym)); end;
if CDTYPE == 'HOT', codes = eye(length(sym)); end;
if ~exist('codes'), error(sprintf('No encoding type (%s)',CDTYPE)); end;
    
OSEQ = codes(:,indseq);

