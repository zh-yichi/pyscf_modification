def ghf_mo_density(mf,i,coords):
  #coords = (x,y,z)  i-th MO, 0-based
  import numpy as np
  nao = len(mf.mo_occ)
  mo_coeff_a = mf.mo_coeff[:int(nao/2),i]
  mo_coeff_b = mf.mo_coeff[int(nao/2):,i]
  mol = mf.mol()
  ao_value = mol.eval_gto("GTOval", [coords])
  mo_coeff_a_mod = np.einsum('i,j',mo_coeff_a.conjugate(),mo_coeff_a)
  mo_coeff_b_mod = np.einsum('i,j',mo_coeff_b.conjugate(),mo_coeff_b)
  mo_den_a = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_a_mod,ao_value[0]))
  mo_den_b = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_b_mod,ao_value[0]))
  mo_den = (mo_den_a+mo_den_b)**0.5
  mo_den = mo_den.real
  return mo_den

mo_density = ghf_mo_density(mf,2,[0,0,0])
print(mo_density)


##ghf complex mo phase##
def ghf_mo_phase(mf,i,coords):
  #coords = (x,y,z)  i-th MO, 0-based
  import numpy as np
  nao = len(mf.mo_occ)
  mo_coeff_a = mf.mo_coeff[:int(nao/2),i]
  mo_coeff_b = mf.mo_coeff[int(nao/2):,i]
  mol = mf.mol()
  ao_value = mol.eval_gto("GTOval", [coords])
  ##real part
  mo_coeff_a_mod_real = np.einsum('i,j',mo_coeff_a.real,mo_coeff_a.real)
  mo_coeff_b_mod_real = np.einsum('i,j',mo_coeff_b.real,mo_coeff_b.real)
  mo_den_a_real = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_a_mod_real,ao_value[0]))
  mo_den_b_real = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_b_mod_real,ao_value[0]))
  mo_den_real = (mo_den_a_real+mo_den_b_real)**0.5
  ##imaginary part
  mo_coeff_a_mod_imag = np.einsum('i,j',mo_coeff_a.imag,mo_coeff_a.imag)
  mo_coeff_b_mod_imag = np.einsum('i,j',mo_coeff_b.imag,mo_coeff_b.imag)
  mo_den_a_imag = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_a_mod_imag,ao_value[0]))
  mo_den_b_imag = np.inner(ao_value[0].conjugate(),np.dot(mo_coeff_b_mod_imag,ao_value[0]))
  mo_den_imag = (mo_den_a_imag+mo_den_b_imag)**0.5
  theta = np.arctan(mo_den_imag/mo_den_real)
  return theta

mo_phase = ghf_mo_phase(mf,1,[0,0,0])
print(mo_phase)