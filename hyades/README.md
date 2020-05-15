

python fit-hyades.py -r 10 1000 --v0 -6.086 45.628 5.517 myfit_10_m_fixed_v0
python fit-hyades.py -r 5 15 --v0 -6.103 45.611 5.532 myfit_5_15_fixed_v0.pickle
python fit-hyades.py -r 10 1000 myfit_10_m.pickle
python fit-hyades.py -r 0 10 myfit_0_10


python hyades-mock-fullmodel.py mock_null.pickle



python fit-hyades.py --exclude-other -r 0 10 hyades_0_10
python fit-hyades.py --exclude-other -r 10 1000 hyades_10_m
python fit-hyades.py --exclude-other -r 10 1000 --v0 -6.08588564 45.62882666  5.51807565 hyades_10_m_fixed_v0

# create result summary table for paper
$ python make-table.py --latex --fit cl hyades_0_10.pickle --fit tails hyades_10_m_fixed_v0.pickle