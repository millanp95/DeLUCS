import cv2
import pickle
import numpy as np

from sklearn.preprocessing import normalize


# ------ add_noise ----------------

def add_noise(x_train):

    row, col = x_train.shape
    
    alpha = 0.75 # weight of the first array elements
    beta = 1 - alpha # weight of the second array elements
    gamma = 0 # scalar added to each sum
    
    #We might want to look into this one. 
    gaussian = np.random.random((row, col, 1)).astype(np.float32)/1000
    gaussian_train = cv2.addWeighted(x_train, alpha, gaussian, beta, gamma)
    
    return np.array(gaussian_train, dtype = np.float32)

# ----- Transition -----------------
def transition(seq, threshold):
  mutated_seq = []
  for nucleotide in seq:
    prob = np.random.uniform()
    if prob > threshold:
      if nucleotide == 'A':
        mutated_seq.append('G')
      if nucleotide == 'G':
        mutated_seq.append('A')
      if nucleotide == 'T':
        mutated_seq.append('C')
      if nucleotide == 'C':
        mutated_seq.append('T')
    else:
      mutated_seq.append(nucleotide)
  l = len(mutated_seq)
  mutated_seq += ' ' * (len(seq) - l)
  return ''.join(mutated_seq)
  
  
# ------ Transversion ---------------

def transversion(seq, threshold):
  mutated_seq = []
  for nucleotide in seq:
    prob = np.random.uniform()
    if prob > threshold:
      if nucleotide == 'A':
        random_number = np.random.uniform()
        if random_number > 0.5:
          mutated_seq.append('T')
        else:
          mutated_seq.append('C')
      if nucleotide == 'G':
        random_number = np.random.uniform()
        if random_number > 0.5:
          mutated_seq.append('T')
        else:
          mutated_seq.append('C')
      if nucleotide == 'T':
        random_number = np.random.uniform()
        if random_number > 0.5:
          mutated_seq.append('A')
        else:
          mutated_seq.append('G')
      if nucleotide == 'C':
        random_number = np.random.uniform()
        if random_number > 0.5:
          mutated_seq.append('A')
        else:
          mutated_seq.append('G')
    else:
      mutated_seq.append(nucleotide)
  l = len(mutated_seq)
  mutated_seq += ' ' * (len(seq) - l)
  return ''.join(mutated_seq)
  
