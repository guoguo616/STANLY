function F = gpdist(image, probes)
% GPDIST compute pdist using geodesic distance
%
%    IMAGE is expected to be a three dimensional voxel image
%    PROBES is expected to be a vector of coordinates
%
%    F = GPDIST(image, probes) is a two dimensional matrix of size
%    n by n where n is the length of PROBES. The (i,j) cell contains
%    the distance from probes[i] to probes [j]

% Error handling

%Check that the image is the correct dimension
dim = size(size(image));
if( dim(2) ~= 3)
    e_message = sprintf("gpdist expects image to be of dimension 3, got dimesion %d", dim(2));
    error(e_message);
end

%Check that probes are of the correct dimension
dim = size(probes);
dim_ = size(dim);
if( dim(2) ~= 3 || dim_(2) ~= 2 )
    error("gpdist expects probes to be of dimension n by 3");
end

%Check if any probes are out of bounds:
[rows, columns] = size(probes);
for row = 1 : rows
    probe = probes(row,:);
    vector = sprintf("%d ", probe);
    vector = sprintf("[ %s]", vector);
    if(any(probe < [1,1,1]))
        error("The probe %s is out of bounds, did you forget to map to voxel space?",vector);
    end
    if(any(probe > size(image)))
        sz = sprintf("%d ", size(image));
        error("The probe %s is out of bounds, the image has size %s",vector, sz);
    end
end

image = 1./double( image );
F = pdist_interop(image, probes);