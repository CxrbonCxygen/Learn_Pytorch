# @Version : 1.0
# @Auther : CarbonOxygen
# @File : CallTest.py
# @Time : 2026/5/7 12:39

class Person:
    def __call__(self, name):
        print("__call__"+ "Hello" + name)

    def hello(self, name):
        print("hello" + name)

# 测试

person = Person()
person("CarbonOxygen")
person.hello("CarbonOxygen")