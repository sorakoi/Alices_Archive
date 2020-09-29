#include<iostream>
#include<string>
using namespace std;
class makura
{public:
	void in();
    void out();
private:
	string tel,qq,name,mails;
};
void makura::in()
{cout<<endl<<"姓名："<<endl;cin>>name;
cout<<"电话："<<endl;cin>>tel;
cout<<"QQ："<<endl;cin>>qq;
cout<<"邮箱：";cin>>mails;
}
void makura::out()
{cout<<"姓名："<<name<<"  电话："<<tel<<"  QQ："<<qq<<"  电子邮箱："<<mails<<endl;}
int main()
{
	makura hime[50];
	int z,i=0,j=0;
	cout<<"首次登录，请先输入通讯录：";
	hime[i].in(); 
	cout<<"请输入指令：";
	for(;;)
	{cin>>z;
	switch(z)
	{case(1):
		i++;hime[i].in();break;
	case(2):
		for(j=0;j<=i;j++)
		{hime[j].out();} break;
//	case(3):
	default:
       cout << "goodbye" << endl;break;
	}}
	return 0;
}
