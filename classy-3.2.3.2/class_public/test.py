import sys
sys.path.insert(0,"/home/nilsor/codes/class_public/class_public/python/build/lib.linux-x86_64-3.8/")

from classy import Class

params={'N_ncdm': 1,
        'N_ur': 2.0328,
        'non linear': 'hmcode', 
        'output': 'mPk', 
        'z_pk':'3',
        'P_k_max_1/Mpc': 15,
        '100*theta_s': 1.0417368567740997, 
        'A_s': 2.1102839631833067e-09, 
        'm_ncdm': 0.06, 
        'n_s': 0.9671828913167801, 
        'omega_b': 0.02255434170350831,
        'omega_cdm': 0.11817158102801589,
        'tau_reio': 0.05515052267600599}

c = Class()

c.set(params)

c.compute()

print("getting pk")

pk = c.get_pk_and_k_and_z()   # this works fine

print("got pk")

c.set({'output':'mPk tCl'})

c.compute()
print("getting pk")

apk = c.get_pk_and_k_and_z()    # this line produces an error

print("got pk")

print(pk,apk)
