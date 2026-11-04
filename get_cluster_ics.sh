# Each cluster is 5e4 Msun. At fixed rho_0 the mass does not touch the
# dimensionless model -- it rescales r_0, r_hm and sigma as M_GC^(1/3) and moves
# the E_J/E_tilde escape diagnostics through the cluster-host mass ratio.
M_GC=5e4

python generate_ics_for_spinning_clusters.py 200000 --psi 3 --rho_0 80 --host mw2022_sph --M_GC $M_GC --r_gal 3 --zeta_max --n_real 5
python generate_ics_for_spinning_clusters.py 200000 --psi 3 --rho_0 80 --host mw2022_sph --M_GC $M_GC --r_gal 10 --zeta_max --n_real 5
python generate_ics_for_spinning_clusters.py 200000 --psi 3 --rho_0 80 --host mw2022_sph --M_GC $M_GC --r_gal 3 --zeta -1 --n_real 5
python generate_ics_for_spinning_clusters.py 200000 --psi 3 --rho_0 80 --host mw2022_sph --M_GC $M_GC --r_gal 10 --zeta -1 --n_real 5

#python3 generate_ics_for_spinning_clusters.py 200000 --psi 3 --rho_0 500 --host mw2022_sph --M_GC $M_GC --r_gal 10 --zeta 0 # corotating for unit test
