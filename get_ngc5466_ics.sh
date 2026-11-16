# TIRO initial conditions for the NGC 5466 KRIOS runs (KRIOS/src/ngc5466_n2e5_config_files): one
# non-spinning control (zeta = -1) and one maximally spinning model (--zeta_max = 0.99 zeta_crit) on
# the catalogue orbit, N = 200000, one realisation each (seed 42). See NGC5466_N2E5_ICS.md.
# NGC 5466 models at any other N (the earlier N = 150000 set) are deprecated.
#
# --host mw2022_sph here sets only the two numbers TIRO takes from a host, the circular frequency
# Omega and the tidal anisotropy nu at --r_gal, which is why the sampler needs a spherical host. The
# KRIOS runs themselves orbit in the FULL MilkyWayPotential2022 (HOST_POTENTIAL_TO_USE=1, disk
# included) on NGC 5466's own eccentric, inclined orbit. r_gal is therefore not where the cluster
# sits: it is the radius whose circular frequency in the spherical part equals the orbit-averaged
# <Omega_c^2>^1/2 of the real orbit over the radial period containing t = -5 Gyr. nu at that radius
# (1.83) is 11% below the orbit's effective value, a mismatch the generator cannot take.
#
# The start is set by hand: M_GC = M_0 = 1.05 x the catalogue's present-day 5.61e4 Msun, and rho_0
# makes the psi = 3 control's r_hm equal 1.05 x the present-day 13.75 pc = 14.44 pc. Every number
# comes from spinning_gcs_paper/ngc5466_krios/plan_ngc5466_n2e5_ics.py
# (outputs/n2e5/ngc5466_n2e5_plan.txt).
#
# Expected output directories (the generator names them from its arguments; the spinning one carries
# 0.99 zeta_crit to one decimal) must match ICS_FILENAME in the inis, under
# the KRIOS run's data/ngc5466_tiro_ics_subdir/:
#   tiro_ics_N200000_psi3.0_zeta-1.0_hostmw2022_sph_nu1.833_mgc5.89e+04_rho08e+00_rgal19.0897
#   tiro_ics_N200000_psi3.0_zeta3.9_hostmw2022_sph_nu1.833_mgc5.89e+04_rho08e+00_rgal19.0897
# Check physical_units.csv before copying: r_hm_pc 14.44 / 14.54 (control / spinning),
# m_gc_phys_msun 5.89e4. Directories tagged mgc5.95e+04 or zeta4.4 are from the deprecated set's
# parameters and must not be used with these inis.
#
# Run with the krios_env python (the Mac tolerates ~2 workers, 1 BLAS thread):
#   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 bash get_ngc5466_ics.sh

set -e

N=200000
PSI=3
R_GAL=19.0897   # catalogue orbit (Vasiliev & Baumgardt 2021 coordinates)
M_GC=5.89e+04
RHO_0=7.518

python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal $R_GAL --M_GC $M_GC --rho_0 $RHO_0 --n_real 1 --zeta -1
python generate_ics_for_spinning_clusters.py $N --psi $PSI --host mw2022_sph --r_gal $R_GAL --M_GC $M_GC --rho_0 $RHO_0 --n_real 1 --zeta_max
