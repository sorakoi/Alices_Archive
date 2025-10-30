# Himari Department of Sumisora University
# Hoshiori Lab. Python Dep.

class Student(object):
    def __init__(self, uid, id, result):
        self.uid = uid
        self.id = id
        self.result = result
    def print_some(self):
        print("student")
yml = Student(1012,'himari',90)
yml.print_some()


# xuesheng类继承了Student类，所以xuesheng类完整拥有Student类的所有属性和函数
class xuesheng(Student):
    pass
ailishi = xuesheng(1013,'alice',85)
ailishi.print_some()  # 继承了Stu类的函数

class primary_stu(Student):
    def print_some(self):
        print("primary")

def onefunction(person):
    person.print_some()   # 实际上我们并不关心实例是不是Student或者其子类的类型，只要你有print_some函数，就都可以调用本函数。这就是所谓的鸭子类型。

print('以下是关于多态的测试')
onefunction(Student(0,0,0))  # 主要是测试多态，参数随便写的。这里没有把实例赋给变量，所以是一次性的
onefunction(xuesheng(1,1,1)) # 虽然xuesheng子类下没有设置printsome函数，但调用了父类的函数
onefunction(primary_stu(2,2,2))  # 可以看到只有本函数的结果不一样，因为程序自动优先调用了子类的printsome函数

# 关于这个多态，反正就是说，自动从下往上找run函数，如果子类定义了就用子类的，如果没有就用父类的run函数。因为子类实例同时拥有子类和父类的数据类型，所以叫多态
