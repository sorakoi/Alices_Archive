#include<iostream>
#include<stdlib.h>
#include<time.h>
using namespace std;
int main()
{
    int i,j,sjs;char ru[3];
    int rus;
	srand((unsigned)time(NULL));  //生成随机数 
	sjs=rand()%1000;
	cout<<"（测试用）随机数的值："<<sjs<<endl;
	cout<<"请输入指令！"<<endl; 
	cin>>i;
	switch(i)
	{case(1):
		for(j=1;;j++)
		{cout<<"请输入数：";cin>>ru;if(ru[0]=='q') break;
		rus=ru[2]-'0'+10*(ru[1]-'0')+100*(ru[0]-'0');        //将char转换为int 
		if(rus==sjs) {cout<<"恭喜猜对！共花费"<<j<<"次";break;}
		if(rus<sjs) cout<<"输入过小！"<<endl;
		if(rus>sjs) cout<<"输入过大！"<<endl; } break;
	case(2):
			for(j=1;;j++)
		{if(j==11) {cout<<"超过10次错误，你已经失败！";break;}
		cout<<"请输入数：";cin>>ru;if(ru[0]=='q') break;
		rus=ru[2]-'0'+10*(ru[1]-'0')+100*(ru[0]-'0');
		if(rus==sjs) {cout<<"恭喜猜对！共花费"<<j<<"次";break;}
		if(rus<sjs) cout<<"输入过小！"<<endl;
		if(rus>sjs) cout<<"输入过大！"<<endl; } break;
	default:
       cout << "指令错误。请重新输入！！！" << endl;break;
	} 
	return 0;
}
