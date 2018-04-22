import cv2
import numpy as np, sys

print (cv2.__version__)
print (sys.version)

A = cv2.imread(sys.path[0] + "/telegram_blue.png")
B = cv2.imread(sys.path[0] + "/telegram_purple.png")

DEPTH = 9

G = A.copy()
gpA = [G]
for i in xrange(DEPTH):
    G = cv2.pyrDown(G)
    gpA.append(G)

G = B.copy()
gpB = [G]
for i in xrange(DEPTH):
    G = cv2.pyrDown(G)
    gpB.append(G)

lpA = [gpA[DEPTH - 1]]
for i in xrange(DEPTH - 1, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

lpB = [gpB[DEPTH - 1]]
for i in xrange(DEPTH - 1, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
    LS.append(ls)

ls_ = LS[0]
for i in xrange(1, DEPTH):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))

cv2.imwrite('Pyramid_blending2.jpg', ls_)
cv2.imwrite('Direct_blending.jpg', real)
