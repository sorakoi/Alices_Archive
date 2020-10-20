#include<iostream>
#include<stdlib.h>
#include<time.h>
using namespace std;
int main()
{
    int i,sjs,ru;
	srand((unsigned)time(NULL));  //生成随机数 
	sjs=rand()%1000;
	cout<<"（测试用）随机数的值："<<sjs<<endl;
	cout<<"请输入指令！"<<endl; 
	cin>>i;
	switch(i)
	{case(1):
		for(i=1;ru!=100;i++)
		{cin>>ru;if(ru==sjs) {cout<<"恭喜猜对！共花费"<<i<<"次";break;}
		if(ru<sjs) cout<<"输入过小！";
		if(ru>sjs) cout<<"输入过大！"; } break;
	case(2):
		for(i=1;ru!=100||i!=11;i++)
		{cin>>ru;if(ru==sjs) {cout<<"恭喜猜对！共花费"<<i<<"次";break;}
		if(ru<sjs) cout<<"输入过小！";
		if(ru>sjs) cout<<"输入过大！"; } break;
	default:
       cout << "指令错误。请重新输入！！！" << endl;break;
	} 
	return 0;
}
