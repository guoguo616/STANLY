function [alignedImage] = alignFiducials(inputImage)
aR = [0, 90, 180, 270];
for i = 1:4
    rotAngle = aR(i);
    act = imrotate(inputImage, rotAngle);
    cul = normxcorr2(ulf, act);% searches for fiducials in each orientation
    C(1, n) = max(cul(:));  % extracts max match for each fiducial
    cur = normxcorr2(urf, act);
    C(2, n) = max(cur(:));
    cll = normxcorr2(llf, act);
    C(3, n) = max(cll(:));
    clr = normxcorr2(lrf, act);
    C(4, n) = max(clr(:));
end
end