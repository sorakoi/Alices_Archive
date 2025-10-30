# himari department of sumisora university
# hoshiori lab. python dep.

# __init__ 在创建对象时自动调用 (类似构造函数)
# self 参数代表对象本身
class User(object):
    def __init__(self, uid , birth , id):
        self.__uid = uid  # 实例的变量名如果以__开头，就变成了一个私有变量（private）
        self.birth = birth
        self.id = id
    def print_info(self):
        print('%s,%s',self.__uid,self.id)
himari = User(1001,2001, 'yangmoli')
print(himari)
himari.print_info()
print(himari.id)
# print(himari.__uid) # 私有变量无法被外部访问，只能通过内部函数来访问
