# workfunction-MXenes

Codes used to estimate the work function of MXenes from the physico-chemical properties of the constituting elements. These codes are part of a paper authored by Pranav Roy, Lavie Rekhi, See Wee Koh, Hong Li,* and Tej S. Choksi* (2023), Journal of Physics: Energy, 5, 034005, DOI 10.1088/2515-7655/acb2f8. Pranav Roy and Lavie Rekhi contributed equally to the work. Hong Li and Tej S. Choksi are the corresponding authors for this work.

The five files used in this publication are:

**1. C2DB-15-Features.py**

This file regresses the work function of MXenes taken from the Computational 2D Materials Database (C2DB), (https://cmr.fysik.dtu.dk/c2db/c2db.html) to 15 features representing elemental properties of MXenes. 

**2. C2DB-10-8-5-Features.py**

This file regresses the work function of MXenes taken from the Computational 2D Materials Database (C2DB), () to reduced-order feature-sets containing 10, 8, and 5 features. A neural network model is used for the regression. 

**3. C2DB-Occurence-Probability.py**

This file calculates the occurrence probability of the most frequently occurring features across different 5-feature neural network models.

**4. C2DB-Symbolic-Transformer.py**

This file uses a symbolic transformer to find meta-features that demonstrate a high Pearson correlation coefficient with the work function of MXenes. 

**5. C2DB FINAL DATA.xlsx**

This file contains the data used by the models.

The work function of MXenes were taken from C2DB (https://cmr.fysik.dtu.dk/c2db/c2db.html). Calculations for additional MXenes, beyond the C2DB dataset are listed in the Supporting Information. Elemental properties are taken from databases cited in the publication. Additional information required to execute these codes can be obtained by writing to the corresponding author, Tej S. Choksi. 
