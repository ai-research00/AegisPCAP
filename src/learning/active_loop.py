class ActiveLoop:
    def __init__(self,t=-0.25): self.t=t; self.buf=[]
    def eval(self,f,s):
        if s<self.t: self.buf.append(f)
    def retrain(self,m):
        if len(self.buf)>50: m.train(self.buf); self.buf=[]
