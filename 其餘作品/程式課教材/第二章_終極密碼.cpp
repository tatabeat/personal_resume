#include <bits/stdc++.h>
using namespace std;
int main()
{
    srand(time(0));
    int number=rand()%1000;
    int high=1000,low=0,t=10;
    for(;;){
        cout<<"�ѤU"<<t<<"���A";t--;
        cout<<"�q"<<low<<"��"<<high<<"����@�ӼƦr:";
        int n;cin>>n;
        if(n==number){
            cout<<"bingo";
            break;
        }
        else if(n>number)high=n;
        else low=n;
        if(t==0){
            cout<<endl<<"���ƥΧ��F"<<endl;
            cout<<"���׬O:"<<number;
            break;
        }
    }
}
