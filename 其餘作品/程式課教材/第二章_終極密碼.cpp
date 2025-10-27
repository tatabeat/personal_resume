#include <bits/stdc++.h>
using namespace std;
int main()
{
    srand(time(0));
    int number=rand()%1000;
    int high=1000,low=0,t=10;
    for(;;){
        cout<<"剩下"<<t<<"次，";t--;
        cout<<"從"<<low<<"到"<<high<<"中選一個數字:";
        int n;cin>>n;
        if(n==number){
            cout<<"bingo";
            break;
        }
        else if(n>number)high=n;
        else low=n;
        if(t==0){
            cout<<endl<<"次數用完了"<<endl;
            cout<<"答案是:"<<number;
            break;
        }
    }
}
