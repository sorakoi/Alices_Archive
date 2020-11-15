#include<iostream>
#include<fstream>//文件流类头文件 
#include<algorithm>//算法头文件，提供了sort函数 
#include<string>
using namespace std;
class Stu//声明且定义了一个学生类 
{public:
	string name,xuehao,age,sex,born,phone,address,email;
}
stu[1000];
int n=0;//存放学生人数 
void display(); 
void write();//函数声明 
void add()//录入函数 
{cout<<"输入学生人数：";
 cin>>n;
 cout<<"请依次输入学生学号，姓名，年龄，性别，出生年月日，家庭住址，电话，email"<<endl;
 for(int i=0;i<n;i++)
 {cin>>stu[i].xuehao>>stu[i].name>>stu[i].age>>stu[i].sex>>stu[i].born>>stu[i].address>>stu[i].phone>>stu[i].email;}
 cout<<"录入学生信息结束！"<<endl; 
}
bool cmp( Stu b,Stu c)//定义一个比较函数 
{return b.age<c.age;}
bool cmp1(Stu b,Stu c)
{return b.age>c.age;}//判断年龄 
bool cmp2(Stu b,Stu c)
{return b.xuehao<c.xuehao;}
bool cmp3(Stu b,Stu c)
{return b.xuehao>c.xuehao;}//判断学号 
bool cmp4(Stu b,Stu c) 
{if(b.name<=c.name)
 return true;
 else 
   return false;}
void sort()
{ cout<<"选择排序方式：0--按照姓名排序；1--按照学号升序排序；2--按照学号降序排序；3--按照年龄升序排序；4--按照年龄降序排序"<<endl;
  int m;
  cin>>m;
  switch(m)
  {case 0:{sort(stu,stu+n,cmp4);//排序函数，形式为sort(start,end,cmp) 
           display();}
		   break;
   case 1:{sort(stu,stu+n,cmp2);
           display();}
		   break;
   case 2: {sort(stu,stu+n,cmp3);
            display();}
			break;
   case 3:{sort(stu,stu+n,cmp);
           display();}
		   break;
   case 4:{sort(stu,stu+n,cmp1);
           display();}
		   break;
   default:cout<<"指令错误！"<<endl;
   } 
}
void display()
{ //sort(stu,stu+n,cmp2);//按学号升序排列的显示函数   为配合排序功能的实现，不运行此函数
 for(int i=0;i<n;i++)
 cout<<"学号："<<stu[i].xuehao<<" 姓名："<<stu[i].name<<" 年龄："<<stu[i].age<<" 性别："
 <<stu[i].sex<<" 生日："<<stu[i].born<<" 地址："<<stu[i].address<<" 电话："<<stu[i].phone<<" 电子邮件："<<stu[i].email<<endl;}
void search()//查询函数 
{cout<<"输入想要查询的信息："<<endl;
 string s;
 cin>>s;
 sort(stu,stu+n,cmp2);
 while(1)
 {
    for(int i=0;i<n;i++)
    if((s==stu[i].xuehao)||(s==stu[i].name)||(s==stu[i].age)||(s==stu[i].sex)||(s==stu[i].born)||(s==stu[i].address)||(s==stu[i].phone)||(s==stu[i].email))
     cout<<"查询成功"<<"学号："<<stu[i].xuehao<<" 姓名："<<stu[i].name<<" 年龄："<<stu[i].age<<" 性别："
	 <<stu[i].sex<<" 生日："<<stu[i].born<<" 地址："<<stu[i].address<<" 电话："<<stu[i].phone<<" 电子邮件："<<stu[i].email<<endl;
    break;} 
}
void delect()//删除函数 
{string m;
 bool f=0;//定义一个布尔值变量，用于判断 
 cout<<"请输入要删除人的相关信息：";
 cin>>m;
 for(int i=0;i<n;i++)
    if((m==stu[i].xuehao)||(m==stu[i].name)||(m==stu[i].age)||(m==stu[i].sex)||(m==stu[i].born)||(m==stu[i].address)||(m==stu[i].phone)||(m==stu[i].email)) 
    {for(int j=i;j<n-1;j++)
     {stu[j].xuehao=stu[j+1].xuehao;
      stu[j].name=stu[j+1].name;
      stu[j].age=stu[j+1].age;
      stu[j].sex=stu[j+1].sex;
      stu[j].born=stu[j+1].born;
      stu[j].address=stu[j+1].address;
      stu[j].phone=stu[j+1].phone;
      stu[j].email=stu[j+1].email;
	 }
	 cout<<"删除成功！";
	 n--;f=1;break; 
	}
	if(f) write();
	else 
	   cout<<"未找到对应学生，删除失败！"<<endl; 
}
void modify()//修改函数 
{ cout<<"请选择要修改的对应学生：";
  string s,h;int q,j;cin>>s;
  for(int i=0;i<n;i++)
    {if((s==stu[i].xuehao)||(s==stu[i].name)||(s==stu[i].age)||(s==stu[i].sex)||(s==stu[i].born)||(s==stu[i].address)||(s==stu[i].phone)||(s==stu[i].email))
     cout<<"学号："<<stu[i].xuehao<<" 姓名："<<stu[i].name<<" 年龄："<<stu[i].age<<" 性别："
	 <<stu[i].sex<<" 生日："<<stu[i].born<<" 地址："<<stu[i].address<<" 电话："<<stu[i].phone<<" 电子邮件："<<stu[i].email<<endl;
     j=i;}
     cout<<"请选择要修改的信息:0>学号 1>姓名 2>年龄 3>性别 4>出生年月日 5>家庭地址 6>电话 7>email"<<endl;
     cout<<"请输入对应数字以及修改的结果：";
     cin>>q>>h;
     switch(q)
	 {case 0:stu[j].xuehao=h;break;
	  case 1:stu[j].name=h;break;
	  case 2:stu[j].age=h;break;
	  case 3:stu[j].sex=h;break;
	  case 4:stu[j].born=h;break;
	  case 5:stu[j].address=h;break;
	  case 6:stu[j].phone=h;break;
	  case 7:stu[j].email=h;break;
	 }
	 cout<<stu[j].xuehao<<" "<<stu[j].name<<" "<<stu[j].age<<" "<<stu[j].sex<<" "<<stu[j].born<<" "<<stu[j].address<<" "<<stu[j].phone<<" "<<stu[j].email<<endl;
     cout<<"修改成功！"<<endl; 
	 write();	  
}
void write()//将学生信息以文件形式保存 
{   sort(stu,stu+n,cmp2);//按学号排序 
	ofstream outfile("stu.dat",ios::binary);//定义文件流时指定参数，具有打开磁盘文件功能 
	if(!outfile)
	{cerr<<"出现致命错误，保存失败！"<<endl;
	 abort();}
	cout<<"Loading..."<<endl; 
	for(int i=0;i<n;i++)
	outfile.write((char*)&stu[i],sizeof(stu[i]));
	outfile.close();//保存完毕后关闭磁盘文件 
}
void kotone()//菜单函数 
{while(1)
 {cout<<endl;
  cout<<"******学生管理信息系统******"<<endl;
  cout<<"1.录入-2.显示全部信息--3.查询-4.删除信息"<<endl;
  cout<<"5.修改信息----6.排序----7.退出程序-----"<<endl;
   int number;
   cout<<"请输入你要选择的功能：";
   cin>>number;
   switch(number)
   {case 1:add();break;
    case 2:display();break;
    case 3:search();break;
    case 4:delect();break;
    case 5:modify();break;
    case 6:sort();break;
    case 7:exit(1);
    default:cout<<"指令错误！"<<endl; 
   }
   }
}
int main()//主函数 
{
 kotone();
 return 0;
 } 
