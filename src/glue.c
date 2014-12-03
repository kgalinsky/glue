#include <lapacke.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

int
glue_dgeqrf (gsl_matrix *a, gsl_vector *tau)
{
  return (LAPACKE_dgeqrf ( LAPACK_ROW_MAJOR, a->size1, a->size2, a->data,
                          a->tda, tau->data));
}

int
glue_dorgqr (gsl_matrix *a, gsl_vector *tau)
{
  return (LAPACKE_dorgqr ( LAPACK_ROW_MAJOR, a->size2, a->size2, a->size2,
                          a->data, a->tda, tau->data));
}

int
glue_dgesvd (char jobu, char jobvt, gsl_matrix* a, gsl_vector* s, gsl_matrix* u,
             gsl_matrix* vt, gsl_vector* superb)
{
  size_t minmn = a->size1 < a->size2 ? a->size1 : a->size2;

  gsl_matrix nullmat =
    { 0, 0, a->size1 + a->size2, NULL };

  if (u == NULL)
    {
      if ((jobu == 'N') || (jobu = 'O'))
        u = &nullmat;
      else
        {
          GSL_ERROR("Null u matrix with incompatible jobu", GSL_EINVAL);
          return (GSL_EINVAL);
        }
    }

  if (vt == NULL)
    {
      if ((jobvt == 'N') || (jobvt = 'O'))
        vt = &nullmat;
      else
        {
          GSL_ERROR("Null vt matrix with incompatible jobvt", GSL_EINVAL);
          return (GSL_EINVAL);
        }
    }

  int alloc_s = 0, alloc_superb = 0;
  if (s == NULL)
    {
      alloc_s = 1;
      s = gsl_vector_alloc (minmn);
    }

  if (superb == NULL)
    {
      alloc_superb = 1;
      superb = gsl_vector_alloc (minmn);
    }

  int ret = LAPACKE_dgesvd (LAPACK_ROW_MAJOR, jobu, jobvt, a->size1, a->size2,
                            a->data, a->tda, s->data, u->data, u->tda, vt->data,
                            vt->tda, superb->data);

  if (alloc_s)
    gsl_vector_free (s);
  if (alloc_superb)
    gsl_vector_free (superb);

  return (ret);
}

int
glue_dgesdd (char jobz, gsl_matrix* a, gsl_vector* s, gsl_matrix* u,
             gsl_matrix* vt)
{
  size_t minmn = a->size1 < a->size2 ? a->size1 : a->size2;
  gsl_matrix nullmat =
    { 0, 0, a->size1 + a->size2, NULL };
  int alloc_s = 0, alloc_u = 0, alloc_vt = 0;

  if (s == NULL)
    {
      alloc_s = 1;
      s = gsl_vector_alloc (minmn);
    }

  if (u == NULL)
    {
      if ((jobz == 'N') || ((jobz == 'O') && (a->size1 > a->size2)))
        u = &nullmat;
      else
        {
          alloc_u = 1;
          if ((jobz == 'A') || (jobz == 'O'))
            u = gsl_matrix_alloc (a->size1, a->size1);
          else if ((jobz == 'S'))
            u = gsl_matrix_alloc (a->size1, minmn);
          else
            GSL_ERROR("All u allocation options exhausted", GSL_ESANITY);
        }
    }

  if (vt == NULL)
    {
      if ((jobz == 'N') || ((jobz == 'O') && (a->size1 < a->size2)))
        vt = &nullmat;
      else
        {
          alloc_vt = 1;
          if ((jobz == 'A') || (jobz == 'O'))
            vt = gsl_matrix_alloc (a->size2, a->size2);
          else if ((jobz == 'S'))
            vt = gsl_matrix_alloc (a->size2, minmn);
          else
            GSL_ERROR("All vt allocation options exhausted", GSL_ESANITY);
        }
    }

  int ret = LAPACKE_dgesdd (LAPACK_ROW_MAJOR, jobz, a->size1, a->size2, a->data,
                            a->tda, s->data, u->data, u->tda, vt->data,
                            vt->tda);

  if (alloc_s)
    gsl_vector_free (s);
  if (alloc_u)
    gsl_matrix_free (u);
  if (alloc_vt)
    gsl_matrix_free (vt);

  return (ret);
}

double
glue_dlange (const char norm, const gsl_matrix* a)
{
  return (LAPACKE_dlange (LAPACK_ROW_MAJOR, norm, a->size1, a->size2, a->data,
                          a->tda));
}
