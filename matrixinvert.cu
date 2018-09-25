/*-------------------------------------------------------------------------
 *
 *      Function:     MatrixInvert
 *      Description:  A general matrix inverter
 *
 *      Arguments:
 *          mat     Memory containing matrix to be converted.  Assumes
 *                  components to be inverted reside in rows zero thru
 *                  order-1 and columns zero thru order-1
 *          invMat  Memory into which the inverted matrix will be
 *                  returned to the caller.
 *          lda     Specifies the leading dimension of the matrices <mat>
 *                  and <invMat>
 *          order   Specifies the order of the matrix being inverted.
 *
 *------------------------------------------------------------------------*/
#include "binary.h"


int MatrixInvert(double *mat, double *invMat, int order, int lda)
{
        int     i, j, k, offset1, offset2, matSize;
        double  tmpVal, tmpMax, fmax, fval, eps = 1.0e-12;
        double  *tmpMat;


        matSize = lda * lda * sizeof(double);
        tmpMat = (double *)calloc(1, matSize);

/*
 *      Allocate a temporary array to help form the augmented matrix
 *      for the system.  Initialize tmpMat to be a copy of the input
 *      matrix and the inverse to the identity matrix.
 */

        for (i = 0; i < order; i++) {
            for (j = 0; j < order; j++) {
                offset1 = i * lda + j;
                invMat[offset1] = (double)(i == j);
                tmpMat[offset1] = mat[offset1];
            }
        }

        for (i = 0; i < order; i++) {

            fmax = fabs(tmpMat[i*lda+i]);
/*
 *          If tmpMat[i][i] is zero, find the next row with a non-zero
 *          entry in column i and switch that row with row i.
 */
            if (fmax < eps) {
                for (j = i+1; j < order; j++) {
                    if ((tmpMax = fabs(tmpMat[j*lda+i])) > fmax) {
                        fmax = tmpMax;
                        for (k = 0; k < order; k++) {
                            offset1 = i * lda + k;
                            offset2 = j * lda + k;
                            tmpVal = tmpMat[offset1];
                            tmpMat[offset1] = tmpMat[offset2];
                            tmpMat[offset2] = tmpVal;
                            tmpVal = invMat[offset1];
                            invMat[offset1] = invMat[offset2];
                            invMat[offset2] = tmpVal;
                        }
                        break;
                    }
                }
            }

/*
 *          If can't do the inversion, return 0
 */
            if (fmax < eps) {
                printf("MatrixInvert(): unable to invert matrix!");
            }

/*
 *          Multiply all elements in row i by the inverse of tmpMat[i][i]
 *          to obtain a 1 in tmpMat[i][i]
 */
            fval = 1.0 / tmpMat[i*lda+i];

            for (j = 0; j < order; j++)   {
                offset1 = i * lda + j;
                tmpMat[offset1] *= fval;
                invMat[offset1] *= fval;
            }

/*
 *          Insure that the only non-zero value in column i is in row i
 */
            for (k = 0; k < order; k++) {
                if (k != i) {
                    fval = tmpMat[k*lda+i];
                    for (j = 0; j < order;  j++) {
                        offset1 = k * lda + j;
                        offset2 = i * lda + j;
                        tmpMat[offset1] -= fval*tmpMat[offset2];
                        invMat[offset1] -= fval*invMat[offset2];
                    }
                }
            }

        }   /* for (i = 0; i < order; ...) */

        free(tmpMat);

        return(1);
}

