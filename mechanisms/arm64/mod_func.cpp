#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _Ca_HVA_reg(void);
extern void _Ih_reg(void);
extern void _Kv3_1_reg(void);
extern void _NaTa_t_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"Ca_HVA.mod\"");
    fprintf(stderr, " \"Ih.mod\"");
    fprintf(stderr, " \"Kv3_1.mod\"");
    fprintf(stderr, " \"NaTa_t.mod\"");
    fprintf(stderr, "\n");
  }
  _Ca_HVA_reg();
  _Ih_reg();
  _Kv3_1_reg();
  _NaTa_t_reg();
}

#if defined(__cplusplus)
}
#endif
