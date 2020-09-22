#include<iostream>
using namespace std;
class makura
{public:
	void in();
    void out();
private:
	string tel,qq,name,mails;
};
void makura::in()
{cout<<endl<<"姓名：";cin>>name;cout<<"电话：";cin>>tel;cout<<"QQ：";cin>>qq;
cout<<"邮箱：";cin>>mails;}
void makura::out()
{cout<<"姓名："<<name<<"  电话："<<tel<<"  QQ："<<qq<<"  电子邮箱："<<mails<<endl;}
int main()
{
	int z,i=0,j=0;
	cout<<"首次登录，请先输入通讯录：";
	makura hime[i];
	hime[i].in(); 
	cout<<"请输入指令：";
	for(i=1;z!=4;)
	{cin>>z;
	makura hime[i];
	switch(z)
	{case(1):
		hime[i].in();i++; 
	case(2):
		for(j=0;j<=i;j++)
		{hime[j].out();} 
//	case(3):
	default:
        cout << "goodbye" << endl;
	}}
	return 0;
}
