#include <stdio.h>
int ans[92][8], n, b, i, j, num, hang[8];
void queen(int i){
	int j, k;
	if(i == 8){ //一组新的解产生了
		for(j = 0; j < 8; j++)  ans[num][j] = hang[j] + 1;
		num++;
		return;
	}
	for (j=0; j<8; j++){ //将当前皇后i逐一尝试放置在不同的列
        for(k=0; k<i; k++) //逐一判定i与前面的皇后是否冲突
            if( hang[k] == j || (k - i) == (hang[k] - j) || (i - k) == (hang[k] - j )) break;
		if (k == i) {  //放置i，尝试第i+1个皇后
			hang[i] = j;
			queen(i + 1);
		}
	}
}
void main( ){
	num=0;
	queen(0);
	scanf(“%d”, &n);
	for(i = 0; i < n; i++){
		scanf(“%d”, &b);
		for(j = 0; j < 8; j++)  printf(“%d”, ans[b - 1][j]);
		printf(“\n”);
	}
}
