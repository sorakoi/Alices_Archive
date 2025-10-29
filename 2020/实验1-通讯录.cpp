#include<iostream>
#include<string>
using namespace std;
class makura
{public:
	void in();
    void out();
    int find(string);
private:
	string tel,qq,name,mails;
};
void makura::in()
{cout<<"姓名：";cin>>name;cout<<"电话：";cin>>tel;
cout<<"QQ：";cin>>qq;cout<<"邮箱：";cin>>mails;cout<<endl;}
void makura::out()
{cout<<"姓名："<<name<<"  电话："<<tel<<"  QQ："<<qq<<"  电子邮箱："<<mails<<endl;}
int makura::find(string a)
{return name.compare(a);}
int main()
{
	makura hime[50];
	int z,i=0,j=0,t=0;
	string findname;
	cout<<"------------输入指南-------------"<<endl<<"输入1：添加通讯录。输入2：显示所有通讯录。"
	<<endl<<"输入3：查找通讯录。输入5直接退出。"<<endl<<endl;
	cout<<"首次登录，请先输入通讯录："<<endl;
	for(z=1;z!=5;)
	{switch(z)
	{case(1):
		hime[i].in();i++;break;
	case(2):
		for(j=0;j<i;j++)
		{hime[j].out();} break;
	case(3):
	    cout<<"请输入要查找的值："; 
		cin>>findname;
		for(j=0;j<i;j++)
		{if (hime[j].find(findname)==0) {t=1;hime[j].out();}}
		if(t!=1) cout<<"找不到结果！"<<endl;t=0;break;
	default:
       cout << "指令错误。请重新输入！！！" << endl;break;
	}
	cout<<"请输入指令：";cin>>z;}
	return 0;
}
