import torch

class KM():
  def __init__(self, w): # w[M][N]
    self.w = w
    self.M, self.N = w.shape
    self.linky = -torch.ones(self.N)
    self.visx = None
    self.visy = None
    self.lx = torch.zeros(self.M) # superscript
    self.ly = torch.zeros(self.N)
    self.lack = 0
    self.inf = 2147483647

  def find(self, x):
      self.visx[x] = True
      for y in range(self.N):
          if self.visy[y]:
              continue
          t = self.lx[x] + self.ly[y] - self.w[x, y]
          if not t:
              self.visy[y] = True
              if (self.linky[y] == -1 or self.find(int(self.linky[y].item()))): # !~linky[y]
                  self.linky[y] = x
                  return True
          else:
              self.lack = min(self.lack, t)
      return False

  def run(self):
      for i in range(self.M):
          for j in range(self.N):
              self.lx[i] = max(self.lx[i], self.w[i, j]); # initialize superscript
      for x in range(self.M):
          while True:
              self.visx = torch.full((self.M,), False, dtype=torch.bool)
              self.visy = torch.full((self.N,), False, dtype=torch.bool)
              self.lack = self.inf
              if self.find(x):
                  break
              for i in range(self.M):
                  if self.visx[i]:
                      self.lx[i] -= self.lack
              for j in range(self.N):
                  if self.visy[j]:
                      self.ly[j] += self.lack
      return self.linky

if __name__ == '__main__':
    tmp = torch.tensor([[ 161,  122, 2, 0, 0, 0, 0],
        [ 19,  22, 90, 0, 0, 0, 0],
        [1, 30, 113, 0, 0, 0, 0],
        [60, 70, 170, 0, 0, 0, 0]])
    result = KM(tmp).run()
