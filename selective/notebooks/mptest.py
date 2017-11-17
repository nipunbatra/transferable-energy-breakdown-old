import multiprocessing as mp

def f(x):
    return x

if __name__ == '__main__':
    pool = mp.Pool()
    results = []
    for i in range(2):
        # print i
        result = pool.apply_async(f, args=(i,))
        results.append(result)
        # print result.get()
        # results[index] = result
    pool.close()
    pool.join()

error = []
for result in results:
    print result.get()
#     error.append(result.get())
# print error