#!/usr/bin/env python3
"""
Experiment #006 Execution Framework
实验6完整执行框架

Purpose: 提供实验006的完整执行和管理框架
包含训练、评估、监控、报告的完整生命周期管理

Usage Examples:
  # 运行完整实验（推荐）
  python run_experiment_006.py --full-experiment
  
  # 运行特定阶段
  python run_experiment_006.py --stage 1
  python run_experiment_006.py --stage 2  
  python run_experiment_006.py --stage 3
  
  # 运行并生成详细报告
  python run_experiment_006.py --full-experiment --detailed-analysis
  
  # 快速验证模式（减少训练步数）
  python run_experiment_006.py --stage 1 --quick-validation
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.logger import setup_logger
from train_experiment_006 import Experiment006Trainer
from evaluate_experiment_006 import Experiment006Evaluator


class Experiment006Framework:
    """
    实验006完整执行框架
    
    提供实验的完整生命周期管理：
    - 实验环境准备
    - 多阶段训练执行
    - 实时监控和状态管理
    - 结果评估和分析
    - 报告生成和存档
    """
    
    def __init__(self, config_path: str = None):
        self.base_dir = Path(__file__).parent
        self.config_path = config_path or str(self.base_dir / "experiment_006_config.json")
        
        # 加载配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.logger = setup_logger("Experiment006Framework")
        
        # 创建实验运行目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # 初始化子系统
        self.trainer = None
        self.evaluator = None
        
        # 执行状态跟踪
        self.execution_status = {
            "start_time": datetime.now().isoformat(),
            "current_stage": None,
            "completed_stages": [],
            "failed_stages": [],
            "overall_success": None
        }
        
        self.logger.info(f"实验006框架初始化完成")
        self.logger.info(f"运行目录: {self.run_dir}")
        self.logger.info(f"配置文件: {self.config_path}")

    def run_full_experiment(self, detailed_analysis: bool = True, 
                          quick_validation: bool = False) -> Dict[str, Any]:
        """
        运行完整的实验006
        
        Args:
            detailed_analysis: 是否进行详细分析
            quick_validation: 是否使用快速验证模式
            
        Returns:
            experiment_results: 完整实验结果
        """
        
        self.logger.info("🚀 开始实验006完整执行")
        self.logger.info("=" * 80)
        
        experiment_results = {
            "experiment_metadata": self.config["experiment_metadata"],
            "execution_config": {
                "detailed_analysis": detailed_analysis,
                "quick_validation": quick_validation,
                "execution_timestamp": self.timestamp,
                "run_directory": str(self.run_dir)
            },
            "stage_results": {},
            "final_analysis": {},
            "execution_summary": {}
        }
        
        try:
            # 1. 环境准备和配置验证
            self._prepare_experiment_environment()
            
            # 2. 创建训练器
            self.trainer = Experiment006Trainer(self.config_path)
            
            # 3. 执行三阶段训练
            symbol = self.config["data_configuration"]["symbol"]
            period = self._get_data_period(quick_validation)
            
            training_results = self.trainer.run_complete_experiment(symbol, period)
            experiment_results["stage_results"] = training_results.get("stage_results", {})
            experiment_results["training_success"] = training_results.get("experiment_success", False)
            
            # 4. 详细分析和评估
            if detailed_analysis:
                self.logger.info("📊 开始详细分析和评估...")
                
                self.evaluator = Experiment006Evaluator(
                    experiment_path=str(self.trainer.experiment_path),
                    reference_experiments=self.config["comparison_and_analysis"]["baseline_experiments"]
                )
                
                evaluation_results = self.evaluator.run_comprehensive_evaluation()
                experiment_results["final_analysis"] = evaluation_results
            
            # 5. 生成执行摘要
            execution_summary = self._generate_execution_summary(experiment_results)
            experiment_results["execution_summary"] = execution_summary
            
            # 6. 保存最终结果
            self._save_experiment_results(experiment_results)
            
            # 7. 生成最终报告
            self._generate_final_report(experiment_results)
            
            self.logger.info("✅ 实验006完整执行成功完成")
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"❌ 实验006执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            experiment_results["execution_error"] = str(e)
            experiment_results["execution_success"] = False
            
            return experiment_results



    def run_single_stage(self, stage_number: int, quick_validation: bool = False) -> Dict[str, Any]:
        """
        运行单个实验阶段
        
        Args:
            stage_number: 阶段编号 (1, 2, 3)
            quick_validation: 是否使用快速验证模式
            
        Returns:
            stage_results: 单阶段结果
        """
        
        self.logger.info(f"🎯 开始执行实验006阶段{stage_number}")
        
        if stage_number not in [1, 2, 3]:
            raise ValueError(f"无效的阶段编号: {stage_number}. 必须是1, 2, 或3")
        
        try:
            # 环境准备
            self._prepare_experiment_environment()
            
            # 创建训练器
            self.trainer = Experiment006Trainer(self.config_path)
            
            # 获取数据配置
            symbol = self.config["data_configuration"]["symbol"]
            period = self._get_data_period(quick_validation)
            
            # 执行指定阶段
            if stage_number == 1:
                stage_result = self.trainer.run_stage1_reward_fix(symbol, period)
            elif stage_number == 2:
                stage_result = self.trainer.run_stage2_forex_specialization(symbol, period)
            elif stage_number == 3:
                stage_result = self.trainer.run_stage3_system_optimization(symbol, period)
            
            # 保存阶段结果
            stage_result_path = self.run_dir / f"stage_{stage_number}_result.json"
            with open(stage_result_path, 'w', encoding='utf-8') as f:
                json.dump(stage_result, f, indent=2, default=str)
            
            self.logger.info(f"✅ 阶段{stage_number}执行完成")
            return stage_result
            
        except Exception as e:
            self.logger.error(f"❌ 阶段{stage_number}执行失败: {e}")
            raise

    def run_analysis_only(self, experiment_path: str) -> Dict[str, Any]:
        """
        仅运行分析和评估（对已完成的实验）
        
        Args:
            experiment_path: 已完成实验的路径
            
        Returns:
            analysis_results: 分析结果
        """
        
        self.logger.info(f"📊 开始分析已完成的实验: {experiment_path}")
        
        try:
            # 创建评估器
            self.evaluator = Experiment006Evaluator(
                experiment_path=experiment_path,
                reference_experiments=self.config["comparison_and_analysis"]["baseline_experiments"]
            )
            
            # 运行综合评估
            analysis_results = self.evaluator.run_comprehensive_evaluation()
            
            # 保存分析结果
            analysis_result_path = self.run_dir / "analysis_only_result.json"
            with open(analysis_result_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            self.logger.info("✅ 分析完成")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ 分析执行失败: {e}")
            raise

    def validate_experiment_setup(self) -> Dict[str, Any]:
        """
        验证实验环境和配置
        
        Returns:
            validation_results: 验证结果
        """
        
        self.logger.info("🔍 验证实验环境和配置...")
        
        validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "config_validation": {},
            "dependency_validation": {},
            "data_validation": {},
            "environment_validation": {},
            "overall_validation": False
        }
        
        try:
            # 1. 配置文件验证
            config_validation = self._validate_configuration()
            validation_results["config_validation"] = config_validation
            
            # 2. 依赖验证
            dependency_validation = self._validate_dependencies()
            validation_results["dependency_validation"] = dependency_validation
            
            # 3. 数据可用性验证
            data_validation = self._validate_data_availability()
            validation_results["data_validation"] = data_validation
            
            # 4. 环境验证
            environment_validation = self._validate_environment()
            validation_results["environment_validation"] = environment_validation
            
            # 5. 整体验证结果
            all_validations = [
                config_validation.get("valid", False),
                dependency_validation.get("valid", False),
                data_validation.get("valid", False),
                environment_validation.get("valid", False)
            ]
            
            validation_results["overall_validation"] = all(all_validations)
            
            if validation_results["overall_validation"]:
                self.logger.info("✅ 实验环境验证通过")
            else:
                self.logger.warning("⚠️ 实验环境验证发现问题")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ 环境验证失败: {e}")
            validation_results["validation_error"] = str(e)
            return validation_results

    def _prepare_experiment_environment(self):
        """准备实验环境"""
        
        self.logger.info("🔧 准备实验环境...")
        
        # 创建必要的目录
        required_dirs = [
            self.run_dir / "logs",
            self.run_dir / "models", 
            self.run_dir / "results",
            self.run_dir / "analysis",
            self.run_dir / "plots"
        ]
        
        for dir_path in required_dirs:
            dir_path.mkdir(exist_ok=True)
        
        # 保存配置副本
        config_copy_path = self.run_dir / "experiment_config_copy.json"
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        self.logger.info("✅ 实验环境准备完成")

    def _get_data_period(self, quick_validation: bool = False) -> str:
        """获取数据周期配置"""
        
        if quick_validation:
            return "3mo"  # 快速验证使用3个月数据
        else:
            return self.config["data_configuration"]["periods"]["stage2_optimization"]

    def _validate_configuration(self) -> Dict[str, Any]:
        """验证配置文件"""
        
        required_sections = [
            "experiment_metadata",
            "data_configuration", 
            "feature_engineering",
            "reward_function_configuration",
            "training_configuration"
        ]
        
        config_validation = {"valid": True, "missing_sections": [], "issues": []}
        
        for section in required_sections:
            if section not in self.config:
                config_validation["missing_sections"].append(section)
                config_validation["valid"] = False
        
        return config_validation

    def _validate_dependencies(self) -> Dict[str, Any]:
        """验证依赖包"""
        
        required_packages = self.config["infrastructure_configuration"]["dependencies"]["key_packages"]
        dependency_validation = {"valid": True, "missing_packages": [], "version_issues": []}
        
        for package_spec in required_packages:
            package_name = package_spec.split(">=")[0]
            try:
                __import__(package_name.replace("-", "_"))
            except ImportError:
                dependency_validation["missing_packages"].append(package_name)
                dependency_validation["valid"] = False
        
        return dependency_validation

    def _validate_data_availability(self) -> Dict[str, Any]:
        """验证数据可用性"""
        
        # 简化的数据验证
        data_validation = {"valid": True, "issues": []}
        
        symbol = self.config["data_configuration"]["symbol"]
        if not symbol:
            data_validation["issues"].append("未指定交易标的")
            data_validation["valid"] = False
        
        return data_validation

    def _validate_environment(self) -> Dict[str, Any]:
        """验证环境设置"""
        
        environment_validation = {"valid": True, "issues": []}
        
        # 检查写入权限
        try:
            test_file = self.run_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            environment_validation["issues"].append(f"写入权限检查失败: {e}")
            environment_validation["valid"] = False
        
        return environment_validation

    def _generate_execution_summary(self, experiment_results: Dict) -> Dict[str, Any]:
        """生成执行摘要"""
        
        summary = {
            "execution_time": self.timestamp,
            "total_stages": len(experiment_results.get("stage_results", {})),
            "successful_stages": 0,
            "failed_stages": 0,
            "overall_success": False,
            "key_achievements": [],
            "main_issues": []
        }
        
        # 分析阶段结果
        for stage_name, stage_result in experiment_results.get("stage_results", {}).items():
            if stage_result.get("success", False):
                summary["successful_stages"] += 1
            else:
                summary["failed_stages"] += 1
        
        # 检查整体成功
        training_success = experiment_results.get("training_success", False)
        analysis_success = experiment_results.get("final_analysis", {}).get("evaluation_success", True)
        
        summary["overall_success"] = training_success and analysis_success
        
        # 提取关键成就
        if training_success:
            summary["key_achievements"].append("多阶段训练成功完成")
        
        if experiment_results.get("final_analysis"):
            summary["key_achievements"].append("详细分析和评估完成")
        
        return summary

    def _save_experiment_results(self, experiment_results: Dict):
        """保存实验结果"""
        
        # 保存完整结果
        results_file = self.run_dir / "complete_experiment_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_results, f, indent=2, default=str, ensure_ascii=False)
        
        # 保存摘要
        summary_file = self.run_dir / "experiment_summary.json"
        summary_data = {
            "experiment_id": "006",
            "timestamp": self.timestamp,
            "success": experiment_results.get("execution_summary", {}).get("overall_success", False),
            "key_metrics": experiment_results.get("execution_summary", {}),
            "run_directory": str(self.run_dir)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"实验结果已保存到: {results_file}")

    def _generate_final_report(self, experiment_results: Dict):
        """生成最终报告"""
        
        report_content = f"""# 实验006执行报告

## 基本信息
- **实验ID**: 006
- **执行时间**: {self.timestamp}
- **运行目录**: {self.run_dir}
- **配置文件**: {self.config_path}

## 实验概览
- **主要目标**: {self.config["experiment_metadata"]["primary_objective"]}
- **次要目标**: {self.config["experiment_metadata"]["secondary_objective"]}
- **预期相关性**: {self.config["experiment_metadata"]["expected_correlation"]}

## 执行摘要
"""
        
        execution_summary = experiment_results.get("execution_summary", {})
        if execution_summary:
            report_content += f"""
- **总阶段数**: {execution_summary.get("total_stages", 0)}
- **成功阶段**: {execution_summary.get("successful_stages", 0)}
- **失败阶段**: {execution_summary.get("failed_stages", 0)}
- **整体成功**: {'✅' if execution_summary.get("overall_success") else '❌'}
"""
        
        # 添加关键成就
        key_achievements = execution_summary.get("key_achievements", [])
        if key_achievements:
            report_content += "\n## 关键成就\n"
            for i, achievement in enumerate(key_achievements, 1):
                report_content += f"{i}. {achievement}\n"
        
        # 添加主要问题
        main_issues = execution_summary.get("main_issues", [])
        if main_issues:
            report_content += "\n## 主要问题\n"
            for i, issue in enumerate(main_issues, 1):
                report_content += f"{i}. {issue}\n"
        
        report_content += f"""
## 详细结果
详细的分析结果和数据文件请查看运行目录:
- 完整结果: `complete_experiment_results.json`
- 阶段结果: `stage_*_result.json`
- 分析图表: `plots/` 目录
- 训练模型: `models/` 目录

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        report_file = self.run_dir / "EXPERIMENT_006_EXECUTION_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"最终报告已生成: {report_file}")


def main():
    """主执行函数"""
    
    parser = argparse.ArgumentParser(
        description="实验006完整执行框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整实验
  python run_experiment_006.py --full-experiment
  
  # 运行特定阶段
  python run_experiment_006.py --stage 1
  
  # 快速验证模式
  python run_experiment_006.py --stage 1 --quick-validation
  
  # 验证环境设置
  python run_experiment_006.py --validate-setup
        """
    )
    
    # 主要选项
    main_group = parser.add_mutually_exclusive_group(required=True)
    main_group.add_argument("--full-experiment", action="store_true",
                           help="运行完整的3阶段实验")
    main_group.add_argument("--stage", type=int, choices=[1, 2, 3],
                           help="运行特定阶段 (1=奖励修复, 2=外汇优化, 3=系统完善)")
    main_group.add_argument("--analysis-only", 
                           help="仅分析已完成的实验（提供实验路径）")
    main_group.add_argument("--validate-setup", action="store_true",
                           help="验证实验环境和配置")
    
    # 可选参数
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--quick-validation", action="store_true",
                       help="使用快速验证模式（减少训练时间）")
    parser.add_argument("--no-detailed-analysis", action="store_true",
                       help="跳过详细分析（仅在full-experiment中有效）")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建框架实例
        framework = Experiment006Framework(args.config)
        
        # 根据参数执行相应操作
        if args.validate_setup:
            print("验证实验环境设置...")
            validation_results = framework.validate_experiment_setup()
            
            print("\n" + "=" * 60)
            print("环境验证结果")
            print("=" * 60)
            
            if validation_results["overall_validation"]:
                print("SUCCESS: 环境验证通过，可以开始实验")
            else:
                print("FAILED: 环境验证失败，请检查以下问题:")
                for section, result in validation_results.items():
                    if isinstance(result, dict) and not result.get("valid", True):
                        print(f"  - {section}: {result}")
        
        elif args.analysis_only:
            print("开始分析已完成的实验...")
            analysis_results = framework.run_analysis_only(args.analysis_only)
            
            print("\n" + "=" * 60) 
            print("分析完成")
            print("=" * 60)
            print(f"分析结果已保存到: {framework.run_dir}")
        
        elif args.stage:
            print(f"开始执行阶段{args.stage}...")
            stage_results = framework.run_single_stage(args.stage, args.quick_validation)
            
            print("\n" + "=" * 60)
            print(f"阶段{args.stage}执行结果")
            print("=" * 60)
            
            success = stage_results.get("success", False)
            print(f"执行状态: {'SUCCESS' if success else 'FAILED'}")
            
            if success:
                val_metrics = stage_results.get("validation_metrics", {})
                if val_metrics:
                    print(f"验证回报: {val_metrics.get('val_mean_return', 0):.2f}%")
                    print(f"验证胜率: {val_metrics.get('val_win_rate', 0):.2f}")
        
        elif args.full_experiment:
            print("开始完整实验...")
            detailed_analysis = not args.no_detailed_analysis
            
            experiment_results = framework.run_full_experiment(
                detailed_analysis=detailed_analysis,
                quick_validation=args.quick_validation
            )
            
            print("\n" + "=" * 80)
            print("实验006完整执行结果")
            print("=" * 80)
            
            execution_summary = experiment_results.get("execution_summary", {})
            overall_success = execution_summary.get("overall_success", False)
            
            print(f"实验状态: {'SUCCESS' if overall_success else 'FAILED'}")
            print(f"总阶段数: {execution_summary.get('total_stages', 0)}")
            print(f"成功阶段: {execution_summary.get('successful_stages', 0)}")
            print(f"失败阶段: {execution_summary.get('failed_stages', 0)}")
            
            # 显示关键成就
            achievements = execution_summary.get("key_achievements", [])
            if achievements:
                print(f"\n关键成就:")
                for achievement in achievements:
                    print(f"   - {achievement}")
            
            # 显示主要问题
            issues = execution_summary.get("main_issues", [])
            if issues:
                print(f"\n主要问题:")
                for issue in issues:
                    print(f"   - {issue}")
            
            print(f"\n详细结果: {framework.run_dir}")
            print("=" * 80)
    
    except KeyboardInterrupt:
        print("\nWARNING: 用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: 执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()