#include <iostream>
using namespace std;
int c[20],n,count=0;
void write()
{
    for(int i=0;i<n;++i)     //i<n 输出行循环 
	    {for(int j=0;j<n;++j)   //输出列循环 
		{if(j==c[i]) cout<<"1 ";   //输出对应结果 
            else cout<<"0 ";}
        cout<<endl;}
    cout<<endl;
}
void find(int r)
    {if(r==n)
	   {write();          //当行数到8时输出 
        ++count; 
        return;}
    for(int i=0;i<n;++i)
	{c[r]=i;
        int t=1;
        for(int j=0; j<r; ++j)
            if(c[r]==c[j]||r-j==c[r]-c[j]||r-j==c[j]-c[r])  //检查行列 
			{t=0;break;}        //用t检测是否出现符合的 
        if(t==1) find(r+1);  //递归 
    }
}
int main()
{
	char b;
	cout<<"此程序可自定义棋盘边长，输入y设置，输入n则默认为8。  "<<"输入值应当在6-10之间！"<<endl;
	cin>>b;
	if(b=='y')
	{cout<<"在这里输入边长：";for(;;){cin>>n;if(n>10||n<6) {cout<<"溢出,请重新输入"<<endl;}else break;}}
	if(b=='n') n=8;
    find(0);
    cout<<"总计共"<<count<<"个符合结果！"<<endl;
    return 0;
}
