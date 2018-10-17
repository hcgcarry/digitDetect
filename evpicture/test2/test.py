import subprocess
lsResult=subprocess.check_output("ls -l",shell=True)
lsResult=lsResult.decode()
lsResultList=lsResult.splitlines()
print(lsResultList)
resultList=[]
successDelet=0

for index,item in enumerate(lsResultList):
    if index!=0:
        item=item.split()
        for item2 in resultList:
            if item2[4]==item[4]:
                if subprocess.check_call('rm {}'.format(item[8]),shell=True)==0:
                    print('success delete {}'.format(item[8]))
                    successDelet=1
                    break
                else:
                    print('rm fault')
                    successDelet=1
        if successDelet==0:
            resultList.append(item)

        successDelet=0

    

