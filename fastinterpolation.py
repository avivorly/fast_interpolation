
import numpy as np
import time
from collections import defaultdict

class FastSearchArray:
    # arr - a sorted array
    # multi - number of hints points for every original point
    def __init__(self, arr, multi=100):
        self._data = np.asarray(arr)
        self.x0 = self._data[0]
        self.xf = self._data[-1]
        self.r = self.xf - self.x0
        l = len(self._data)
        self.num = int(l * multi)
        self.ratio = self.num / self.r
        # bulding the hint array
        values = np.linspace(self.x0, self.x0 + self.r, num=self.num+1, endpoint=True)
        self.hint = np.searchsorted(self._data, values).astype(int)
        self.hint[-1] = self.hint[-2]

        # testing the hint array
        test_points = np.linspace(self.x0,self.xf,self.num*100)
        ground_true = np.searchsorted(arr,test_points)

        test_values = self.hint[np.array(self.ratio * (test_points - self.x0),dtype=int)]
        cancel = test_points[ground_true!=test_values]
        # marking failed test points
        self.hint[np.array(self.ratio * (cancel - self.x0),dtype=int)] = -1

        print(f'hint valid persentage {np.count_nonzero(self.hint==-1)/self.hint.size}')

    def hintsearch(self, p):
      res = self.hint[int(self.ratio * (p - self.x0))]
      if res ==-1:
        return np.searchsorted(self._data,p)
      return res



if __name__ == "__main__":
    n = 3
    start_x = 0.0
    end_x   = 10.0
    num     = 10**n

    xs = np.sort(np.random.uniform(low=start_x, high=end_x, size=num))
    xs[0] = start_x; xs[-1] = end_x

    arr = FastSearchArray(xs)

    times = defaultdict(lambda: 0)

    test_points = np.sort(np.random.uniform(low=start_x, high=end_x, size=30000))


    # 3) Evaluate and measure time
    for x0 in test_points:
        # -- A) Direct NumPy searchsorted
        start_time = time.time()
        val_searchsorted = np.searchsorted(xs, x0)
        end_time = time.time()
        times['searchsorted'] += (end_time - start_time)

        # -- B) Custom hintsearch
        start_time = time.time()
        val_hint = arr.hintsearch(x0)
        end_time = time.time()
        times['hintsearch'] += (end_time - start_time)

        if val_searchsorted - val_hint > 1e-8:
            print(f"Warning: mismatch at x0={x0} -> searchsorted={val_searchsorted}, hintsearch={val_hint}")

    # 4) Average times per interpolation call in case we want to print it
    n_tests = len(test_points)
    for k in times.keys():
        times[k] /= n_tests

    speedup = (times['searchsorted'] / times['hintsearch']
               if times['hintsearch'] != 0 else float('inf'))

    print("Speedup (searchsorted / hintsearch):", speedup)
