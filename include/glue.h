/**
 * @file glue.h
 * @brief GSL L Connect GSL and LAPACK. Currently using lapacke in order to make
 * everything work.
 * TODO: Figure out a backronym for GLUE
 * G GSL
 * L LAPACK
 * U Unifying?
 * E Environment?
 */

#ifndef GLUE_H_
#define GLUE_H_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

int
glue_dgeqrf (gsl_matrix *m, gsl_vector *tau);

int
glue_dorgqr (gsl_matrix *m, gsl_vector *tau);

/**
 * SVD (A=USV'). s/u/vt/superb can all be NULL. u/vt require jobu/vt to be 'Z' or 'O'
 * @param jobu
 * @param jobvt
 * @param a input matrix
 * @param s singular values
 * @param u left singular vectors
 * @param vt right singular vectors
 * @param superb
 * @return success/error code
 */

int
glue_dgesvd (char jobu, char jobvt, gsl_matrix* a, gsl_vector* s, gsl_matrix* u,
             gsl_matrix* vt, gsl_vector* superb);

/**
 * Faster SVD. s/u/vt can all be NULL. Unlike SVD, u/vt can always be NULL.
 * Currently not working where jobz='O' and NULL is provided for u/vt.
 * @param jobz
 * @param a input matrix
 * @param s singular values
 * @param u left singular vectors
 * @param vt right singular vectors
 * @return success/error code
 */

int
glue_dgesdd (char jobz, gsl_matrix* a, gsl_vector* s, gsl_matrix* u,
             gsl_matrix* vt);

double
glue_dlange (const char norm, const gsl_matrix* a);

#endif /* GLUE_H_ */
